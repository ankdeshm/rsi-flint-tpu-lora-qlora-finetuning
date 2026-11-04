#!/usr/bin/env python3
"""Enable static-shape compiled Multi-LoRA (`--lora-backend torch_native`) on TPU in SGLang.

Why this patch is needed on `sglang-torchtpu-diffusion:v0.2.0`:
1. `sglang/srt/lora/layers.py` calls `.pin_memory()` / `pin_memory=True` on CPU offset tensors,
   which is unsupported under `torch_tpu` PJRT CPU allocators.
2. `TpuGraphRunner.can_run_graph()` intentionally returns `False` because TPU HLO caching
   dispatches through `model.forward(...)` compiled by `DecoderTpuCompilePolicy` (`torch.compile(backend="tpu")`).
   Consequently `LoRAManager.prepare_lora_batch()` passes `use_cuda_graph=False` to `TorchNativeLoRABackend`,
   which otherwise falls back to dynamic Python loop/slice `lora_ops.py` (`.item()` calls + variable segment lengths)
   and resets `batch_info = None` during `TpuGraphRunner` precompile warmup.
3. This patch equips `TorchNativeLoRABackend` on TPU (`device.type == "tpu"`) with persistent static
   device buffers (`_tpu_weight_indices` filled with `-1` for base/padding tokens and `_tpu_scalings`),
   keeps `batch_info` active during `TpuGraphRunner` precompile warmup so `apply_lora` is compiled into every
   token bucket once at startup, and executes pure functional static-shape masked `torch.mm` LoRA ops with
   `@torch.compiler.assume_constant_result` slice offsets (zero graph breaks, zero runtime recompilations
   across arbitrary mixes of base + up to `max_loras_per_batch` concurrent adapters).
"""

from __future__ import annotations

import site
from pathlib import Path

PATCH_MARKER = "# [flint-tpu-multilora-static-graph-patch]"

TORCH_BACKEND_TPU_APPEND = f"""

{PATCH_MARKER}
@torch.compiler.assume_constant_result
def _tpu_const_offsets(offset_cpu: torch.Tensor) -> tuple[int, ...]:
    return tuple(int(v) for v in offset_cpu.tolist())


def _tpu_static_lora_a(
    inputs: torch.Tensor,
    weights: torch.Tensor,
    token_weight_indices: torch.Tensor,
    scalings: torch.Tensor,
    num_slices: int = 1,
) -> torch.Tensor:
    total_seq_len = inputs.shape[0]
    if weights.numel() == 0:
        return torch.zeros((total_seq_len, 0), dtype=inputs.dtype, device=inputs.device)
    num_loras, weight_out_dim, _ = weights.shape
    idx = token_weight_indices[:total_seq_len]
    out = torch.zeros((total_seq_len, weight_out_dim), dtype=inputs.dtype, device=inputs.device)
    for lora_idx in range(num_loras):
        mask = (idx == lora_idx).unsqueeze(1)
        x_masked = torch.where(mask, inputs, 0)
        w_a = weights[lora_idx].to(inputs.dtype)
        scale = scalings[lora_idx].to(inputs.dtype)
        out = out + torch.mm(x_masked, w_a.t()) * scale
    return out


def _tpu_static_lora_b(
    inputs: torch.Tensor,
    weights: torch.Tensor,
    token_weight_indices: torch.Tensor,
    offsets: tuple[int, ...],
    base_output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    total_seq_len, input_dim = inputs.shape
    total_output_dim = offsets[-1] if len(offsets) > 0 else 0
    if weights.numel() == 0 or total_output_dim == 0:
        if base_output is not None:
            return base_output
        return torch.zeros((total_seq_len, total_output_dim), dtype=inputs.dtype, device=inputs.device)
    num_loras = weights.shape[0]
    num_slices = len(offsets) - 1
    max_rank = input_dim // num_slices
    idx = token_weight_indices[:total_seq_len]
    slice_deltas = [
        torch.zeros((total_seq_len, offsets[s + 1] - offsets[s]), dtype=inputs.dtype, device=inputs.device)
        for s in range(num_slices)
    ]
    for lora_idx in range(num_loras):
        mask = (idx == lora_idx).unsqueeze(1)
        in_masked = torch.where(mask, inputs, 0)
        for s in range(num_slices):
            x_s = in_masked[:, s * max_rank : (s + 1) * max_rank]
            w_s = weights[lora_idx, offsets[s] : offsets[s + 1], :].to(inputs.dtype)
            slice_deltas[s] = slice_deltas[s] + torch.mm(x_s, w_s.t())
    delta = slice_deltas[0] if num_slices == 1 else torch.cat(slice_deltas, dim=-1)
    if base_output is not None:
        return base_output + delta.to(base_output.dtype)
    return delta


_ORIG_TORCH_NATIVE_INIT = TorchNativeLoRABackend.__init__
_ORIG_TORCH_NATIVE_PREPARE = TorchNativeLoRABackend.prepare_lora_batch
_ORIG_TORCH_NATIVE_A = TorchNativeLoRABackend.run_lora_a_sgemm
_ORIG_TORCH_NATIVE_B = TorchNativeLoRABackend.run_lora_b_sgemm
_ORIG_TORCH_NATIVE_QKV = TorchNativeLoRABackend.run_qkv_lora
_ORIG_TORCH_NATIVE_GATE_UP = TorchNativeLoRABackend.run_gate_up_lora


def _tpu_torch_native_init(self, max_loras_per_batch: int, device: torch.device, **kwargs):
    _ORIG_TORCH_NATIVE_INIT(self, max_loras_per_batch, device, **kwargs)
    self._is_tpu_static = (device.type == "tpu")
    if self._is_tpu_static:
        max_toks = 147456
        self._tpu_max_tokens = max_toks
        self._tpu_weight_indices = torch.full((max_toks,), -1, dtype=torch.int32, device=device)
        self._tpu_scalings = torch.zeros((max_loras_per_batch,), dtype=torch.float32, device=device)
        self._tpu_lora_ranks = torch.zeros((max_loras_per_batch,), dtype=torch.int32, device=device)
        self._tpu_static_batch_info = TorchNativeLoRABatchInfo(
            use_cuda_graph=True,
            bs=1,
            num_segments=max_loras_per_batch,
            max_len=1,
            seg_lens=torch.ones((1,), dtype=torch.int32, device=device),
            seg_indptr=torch.zeros((2,), dtype=torch.int32, device=device),
            weight_indices=self._tpu_weight_indices,
            lora_ranks=self._tpu_lora_ranks,
            scalings=self._tpu_scalings,
            permutation=None,
        )
        self._tpu_static_batch_info.has_active_lora = True
        self.batch_info = self._tpu_static_batch_info


def _tpu_reset_batch_state(self):
    if getattr(self, "_is_tpu_static", False):
        self._tpu_weight_indices.fill_(-1)
        self._tpu_scalings.zero_()
        self._tpu_lora_ranks.zero_()
        self._tpu_static_batch_info.has_active_lora = True
        self.batch_info = self._tpu_static_batch_info
    else:
        self.batch_info = None


def _tpu_prepare_lora_batch(
    self,
    forward_batch: ForwardBatch,
    weight_indices: list[int],
    lora_ranks: list[int],
    scalings: list[float],
    use_cuda_graph: bool,
    use_prefill_cuda_graph: bool = False,
):
    if not getattr(self, "_is_tpu_static", False):
        return _ORIG_TORCH_NATIVE_PREPARE(
            self, forward_batch, weight_indices, lora_ranks, scalings, use_cuda_graph, use_prefill_cuda_graph
        )
    seq_lens_cpu = generate_sequence_lengths(forward_batch, device="cpu").to(torch.int64)
    # Map base-model slots (rank == 0) to -1 so base tokens incur zero adapter contribution
    mapped_wi = [
        (wi if (0 <= wi < len(lora_ranks) and lora_ranks[wi] > 0) else -1)
        for wi in weight_indices
    ]
    if len(mapped_wi) > 0 and seq_lens_cpu.numel() == len(mapped_wi):
        tok_wi_cpu = torch.repeat_interleave(
            torch.tensor(mapped_wi, dtype=torch.int32, device="cpu"),
            seq_lens_cpu,
        )
    else:
        tok_wi_cpu = torch.empty((0,), dtype=torch.int32, device="cpu")
    n_tok = min(int(tok_wi_cpu.numel()), self._tpu_max_tokens)
    full_wi_cpu = torch.full((self._tpu_max_tokens,), -1, dtype=torch.int32, device="cpu")
    if n_tok > 0:
        full_wi_cpu[:n_tok] = tok_wi_cpu[:n_tok]
    self._tpu_weight_indices.copy_(full_wi_cpu, non_blocking=False)

    ranks_cpu = torch.zeros((self.max_loras_per_batch,), dtype=torch.int32, device="cpu")
    scales_cpu = torch.zeros((self.max_loras_per_batch,), dtype=torch.float32, device="cpu")
    for idx_slot in range(min(len(lora_ranks), self.max_loras_per_batch)):
        ranks_cpu[idx_slot] = int(lora_ranks[idx_slot])
        scales_cpu[idx_slot] = float(scalings[idx_slot]) if lora_ranks[idx_slot] > 0 else 0.0
    self._tpu_lora_ranks.copy_(ranks_cpu, non_blocking=False)
    self._tpu_scalings.copy_(scales_cpu, non_blocking=False)

    self._tpu_static_batch_info.bs = forward_batch.batch_size
    self._tpu_static_batch_info.has_active_lora = True
    self.batch_info = self._tpu_static_batch_info


def _tpu_run_lora_a_sgemm(self, x: torch.Tensor, weights: torch.Tensor, stack_num: int = 1, *args, **kwargs) -> torch.Tensor:
    if getattr(self, "_is_tpu_static", False):
        return _tpu_static_lora_a(x, weights, self._tpu_weight_indices, self._tpu_scalings, num_slices=stack_num)
    return _ORIG_TORCH_NATIVE_A(self, x, weights, stack_num=stack_num, *args, **kwargs)


def _tpu_run_lora_b_sgemm(
    self, x: torch.Tensor, weights: torch.Tensor, output_offset_cpu: torch.Tensor, base_output: torch.Tensor = None, *args, **kwargs
) -> torch.Tensor:
    if getattr(self, "_is_tpu_static", False):
        offsets = _tpu_const_offsets(output_offset_cpu)
        return _tpu_static_lora_b(x, weights, self._tpu_weight_indices, offsets, base_output=base_output)
    return _ORIG_TORCH_NATIVE_B(self, x, weights, output_offset_cpu=output_offset_cpu, base_output=base_output, *args, **kwargs)


def _tpu_run_qkv_lora(
    self,
    x: torch.Tensor,
    qkv_lora_a: torch.Tensor,
    qkv_lora_b: torch.Tensor,
    output_offset: torch.Tensor,
    output_offset_cpu: torch.Tensor,
    max_qkv_out_dim: int,
    base_output: torch.Tensor = None,
    n_slices: int = 3,
    *args,
    **kwargs,
) -> torch.Tensor:
    if getattr(self, "_is_tpu_static", False):
        a_out = _tpu_static_lora_a(x, qkv_lora_a, self._tpu_weight_indices, self._tpu_scalings, num_slices=n_slices)
        offsets = _tpu_const_offsets(output_offset_cpu)
        return _tpu_static_lora_b(a_out, qkv_lora_b, self._tpu_weight_indices, offsets, base_output=base_output)
    return _ORIG_TORCH_NATIVE_QKV(
        self, x, qkv_lora_a, qkv_lora_b, output_offset, output_offset_cpu, max_qkv_out_dim, base_output=base_output, n_slices=n_slices, *args, **kwargs
    )


def _tpu_run_gate_up_lora(
    self,
    x: torch.Tensor,
    gate_up_lora_a: torch.Tensor,
    gate_up_lora_b: torch.Tensor,
    output_offset_cpu: torch.Tensor,
    base_output: torch.Tensor = None,
    *args,
    **kwargs,
) -> torch.Tensor:
    if getattr(self, "_is_tpu_static", False):
        offsets = _tpu_const_offsets(output_offset_cpu)
        n_slices = len(offsets) - 1
        a_out = _tpu_static_lora_a(x, gate_up_lora_a, self._tpu_weight_indices, self._tpu_scalings, num_slices=n_slices)
        return _tpu_static_lora_b(a_out, gate_up_lora_b, self._tpu_weight_indices, offsets, base_output=base_output)
    return _ORIG_TORCH_NATIVE_GATE_UP(self, x, gate_up_lora_a, gate_up_lora_b, output_offset_cpu, base_output=base_output, *args, **kwargs)


TorchNativeLoRABackend.__init__ = _tpu_torch_native_init
TorchNativeLoRABackend.reset_batch_state = _tpu_reset_batch_state
TorchNativeLoRABackend.prepare_lora_batch = _tpu_prepare_lora_batch
TorchNativeLoRABackend.run_lora_a_sgemm = _tpu_run_lora_a_sgemm
TorchNativeLoRABackend.run_lora_b_sgemm = _tpu_run_lora_b_sgemm
TorchNativeLoRABackend.run_qkv_lora = _tpu_run_qkv_lora
TorchNativeLoRABackend.run_gate_up_lora = _tpu_run_gate_up_lora
"""


def apply_sglang_tpu_lora_patches() -> None:
    sp_dirs = [Path(p) for p in site.getsitepackages()]
    sp = next((p for p in sp_dirs if (p / "sglang" / "srt" / "lora").exists()), Path("/usr/local/lib/python3.12/site-packages"))
    lora_dir = sp / "sglang" / "srt" / "lora"

    # 1. Strip unsupported CPU pin_memory calls in layers.py
    layers_py = lora_dir / "layers.py"
    ltxt = layers_py.read_text()
    if PATCH_MARKER not in ltxt:
        ltxt_new = (
            f"{PATCH_MARKER}\n"
            + ltxt.replace("pin_memory=True", "pin_memory=False").replace(".pin_memory()", "")
        )
        layers_py.write_text(ltxt_new)
        print(f"[patch-lora] Patched {layers_py} (disabled CPU pin_memory for TPU)", flush=True)

    # 2. Ensure mem_pool.py sets pin_memory_available = False on TPU
    mem_pool_py = lora_dir / "mem_pool.py"
    mtxt = mem_pool_py.read_text()
    if PATCH_MARKER not in mtxt:
        old_mp = "self.pin_memory_available = is_pin_memory_available()"
        new_mp = "self.pin_memory_available = False if str(self.device).startswith('tpu') else is_pin_memory_available()"
        if old_mp in mtxt:
            mtxt = mtxt.replace(old_mp, new_mp, 1)
        mem_pool_py.write_text(f"{PATCH_MARKER}\n" + mtxt)
        print(f"[patch-lora] Patched {mem_pool_py} (pin_memory_available=False on TPU)", flush=True)

    # 3. Equip TorchNativeLoRABackend with TPU static-graph multi-LoRA execution
    tb_py = lora_dir / "backend" / "torch_backend.py"
    tb_txt = tb_py.read_text()
    if PATCH_MARKER not in tb_txt:
        tb_py.write_text(tb_txt + TORCH_BACKEND_TPU_APPEND)
        print(f"[patch-lora] Patched {tb_py} (static-shape TPU compiled TorchNativeLoRABackend)", flush=True)

    # 4. Ensure LoRAManager keeps batch_info active even when has_active_lora is False on TPU static path
    lm_py = lora_dir / "lora_manager.py"
    lm_txt = lm_py.read_text()
    if PATCH_MARKER not in lm_txt:
        old_line = "self.lora_backend.batch_info.has_active_lora = any(\n            lora_ranks[wi] > 0 for wi in weight_indices\n        )"
        new_line = (
            "self.lora_backend.batch_info.has_active_lora = True if getattr(self.lora_backend, '_is_tpu_static', False) "
            "else any(lora_ranks[wi] > 0 for wi in weight_indices)"
        )
        if old_line in lm_txt:
            lm_txt = lm_txt.replace(old_line, new_line, 1)
        lm_py.write_text(f"{PATCH_MARKER}\n" + lm_txt)
        print(f"[patch-lora] Patched {lm_py} (constant has_active_lora=True on TPU static path)", flush=True)


if __name__ == "__main__":
    apply_sglang_tpu_lora_patches()
