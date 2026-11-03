"""LoRA and QLoRA (INT8 & NF4) fine-tuning recipes for RecursiveSI/flint (39.8B MoE) on TorchTPU.

Model: RecursiveSI/flint (39.81B total / 2.52B active MoE, 49 layers:
  - Layers 0..1: dense SwiGLU FFN (dim=2048, hidden=6144) + Full NoPE GQA + per-head sinks
  - Layers 2..48: 256 routed top-6 sigmoid experts (hidden=512) + 1 ungated shared expert (hidden=1024)
  - Hybrid Attention: 35 SWA (window=513, RoPE) + 14 Full Attention (NoPE) + learned softmax sinks)

Recipes (`--module=flint_lora_tpu_recipes --config=<name>`):
  * `flint_40b_lora_bf16`       -- Frozen BF16 base (39.81B) + LoRA (r=16, alpha=32) on wq,wk,wv,wo,w1,w2,w3
  * `flint_40b_qlora_int8`      -- Frozen INT8 base (39.48B quantized: 38.59B TpuGroupedExperts + 0.89B Linears)
                                   all-gathered over ICI as 1-byte int8 shards + Pallas grouped matmul
  * `flint_40b_qlora_nf4`       -- Frozen NF4 base (39.48B quantized at 4.125 bits/param: NF4 block-64 + 8-bit
                                   double-quantized absmax) all-gathered over ICI as packed uint8 shards
  * `flint_40b_lora_bf16_r64`   -- High-rank LoRA ablation (r=64, alpha=128)
  * `flint_40b_lora_bf16_attn`  -- Attention-only LoRA ablation (r=16, alpha=32, wq,wk,wv,wo)
  * `flint_debug_lora_bf16`, `flint_debug_qlora_int8`, `flint_debug_qlora_nf4` -- 8-step smoke tests

Exports TWO HuggingFace PEFT adapter bundles at the end of every run:
  1. `peft_adapter`      (full adapter: self_attn.{q,k,v,o}_proj + mlp.{gate,up,down}_proj + mlp.shared_expert.*)
  2. `peft_adapter_attn` (attention-only slice: self_attn.{q,k,v,o}_proj across all 49 layers for universal
                          multi-LoRA serving compatibility in SGLang-TorchTPU & vLLM-TorchTPU)
"""

from __future__ import annotations

import concurrent.futures
import dataclasses
import datetime as _dt
import gc
import json
import os
import re
import shutil
import time
from typing import Any

# IMPORTANT: import torch BEFORE torchtitan.components.lora / spmd_types to avoid circular backend init
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchtitan.components.checkpointer.dcp import CheckpointManager
from torchtitan.components.data.dataset import SingleDatasetConfig
from torchtitan.components.data.loader import GrainDataLoader
from torchtitan.components.data.packing import ConcatThenSplitPackingConfig
from torchtitan.components.data.sources import HuggingFaceRandomAccessSource
from torchtitan.components.lora import LoRAConverter
from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer.lr_scheduler import LRSchedulersContainer
from torchtitan.components.optimizer.optimizer import default_adamw
from torchtitan.components.validate import Validator
from torchtitan.config.configs import CommConfig, ParallelismConfig, TrainingConfig
from torchtitan.distributed.activation_checkpoint import FullAC
from torchtitan.experiments.tpu.examples.flint import model_registry as flint_model_registry
from torchtitan.experiments.tpu.examples.flint.model import FlintSinkAttention
from torchtitan.experiments.tpu.grouped_experts import TpuGroupedExperts, TpuGroupedExpertsConverter
from torchtitan.experiments.tpu.kernels.expert_segment_matmul import segment_matmul_pallas_2d
from torchtitan.experiments.tpu.kernels.paired_segment_matmul import paired_segment_matmul_pallas_2d
from torchtitan.experiments.tpu.loss import PallasCrossEntropyLoss
from torchtitan.experiments.tpu.token_dispatcher import TpuTokenDispatcherConverter
from torchtitan.experiments.tpu.tpu_job_config import TPUConfig, TpuTrainerCompileConfig, TPUTrainerConfig
from torchtitan.hf_datasets.text_datasets import ChatProcessor

__all__ = [
    "flint_40b_lora_bf16",
    "flint_40b_qlora_int8",
    "flint_40b_qlora_nf4",
    "flint_40b_lora_bf16_r64",
    "flint_40b_lora_bf16_attn",
    "flint_40b_lora_bf16_16k",
    "flint_40b_qlora_int8_16k",
    "flint_40b_lora_bf16_32k",
    "flint_40b_qlora_int8_32k",
    "flint_40b_lora_bf16_64k",
    "flint_40b_qlora_int8_64k",
    "flint_debug_lora_bf16",
    "flint_debug_qlora_int8",
    "flint_debug_qlora_nf4",
]

FLINT_HF_SHM = "/dev/shm/flint_hf"
FLINT_HF_GCS = "/models/models/RecursiveSI/flint"
DEFAULT_DATA = "/models/lora-ft/data/alpaca_cleaned_chat.jsonl"
_FUSED_LOSS_OVERRIDE = "torchtitan.experiments.tpu.overrides.fused_loss.fused_loss_lm_head"

# Note: In QKVLinear.Config, `wkv` builds both `self.wk` and `self.wv`!
LORA_TARGETS_ALL = ["wq", "wkv", "wo", "w1", "w2", "w3"]
LORA_TARGETS_ATTN = ["wq", "wkv", "wo"]

TT2HF_FLINT = {
    "attention.qkv_linear.wq": "self_attn.q_proj",
    "attention.qkv_linear.wk": "self_attn.k_proj",
    "attention.qkv_linear.wv": "self_attn.v_proj",
    "attention.wo": "self_attn.o_proj",
    "feed_forward.w1": "mlp.gate_proj",
    "feed_forward.w3": "mlp.up_proj",
    "feed_forward.w2": "mlp.down_proj",
    "moe.shared_experts.w1": "mlp.shared_expert.gate_proj",
    "moe.shared_experts.w3": "mlp.shared_expert.up_proj",
    "moe.shared_experts.w2": "mlp.shared_expert.down_proj",
}

_RUN: dict[str, Any] = {
    "config": None,
    "quant": None,
    "rank": 16,
    "alpha": 32.0,
    "targets": list(LORA_TARGETS_ALL),
    "hf_prefix": "base_model.model.model.layers",
    "base_model": "RecursiveSI/flint",
    "stats": {},
}


def _rank() -> int:
    return int(os.environ.get("RANK", "0"))


def _log(msg: str) -> None:
    if _rank() == 0:
        print(f"[flint-lora-ft] {msg}", flush=True)


def _resolve_hf_path() -> str:
    if os.path.exists(os.path.join(FLINT_HF_SHM, "model.safetensors.index.json")):
        return FLINT_HF_SHM
    return FLINT_HF_GCS


def _device_mem_gib() -> dict[str, Any]:
    try:
        from torchtitan.tools.utils import device_module

        st = device_module.memory_stats()
        g = lambda k: round(st.get(k, -1) / 2**30, 3)  # noqa: E731
        return {
            "allocated_gib": g("allocated_bytes.all.current"),
            "reserved_gib": g("reserved_bytes.all.current"),
            "peak_allocated_gib": g("allocated_bytes.all.peak"),
            "peak_reserved_gib": g("reserved_bytes.all.peak"),
        }
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


# --------------------------------------------------------------------------------------------
# Dtype policy: frozen base weights in BF16 (then INT8/NF4 if QLoRA); LoRA master weights FP32.
# --------------------------------------------------------------------------------------------
def _cast_frozen_params_(model: nn.Module, dtype: torch.dtype) -> None:
    for mod in model.modules():
        for name, p in list(mod.named_parameters(recurse=False)):
            if p is None or p.requires_grad or not p.is_floating_point() or p.dtype == dtype:
                continue
            mod._parameters[name] = nn.Parameter(p.data.to(dtype), requires_grad=False)


def _noop_init(_p: torch.Tensor) -> None:
    return None


def _wrap_parallelize(orig_fn):
    def parallelize_with_frozen_bf16(model, *args, **kwargs):
        _cast_frozen_params_(model, torch.bfloat16)
        n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_all = sum(p.numel() for p in model.parameters())
        _RUN["stats"]["trainable_params"] = int(n_tr)
        _RUN["stats"]["total_params"] = int(n_all)
        _log(f"params: total={n_all:,} trainable(LoRA)={n_tr:,} ({100.0 * n_tr / max(n_all, 1):.4f}%)")
        if _RUN.get("skip_frozen_init"):
            n_skip = 0
            for mod in model.modules():
                own = list(mod.named_parameters(recurse=False))
                if own and all(not p.requires_grad for _, p in own) and hasattr(mod, "_param_init"):
                    mod._param_init = {n: _noop_init for n, _ in own}
                    n_skip += 1
            _log(f"skipping random init of {n_skip} frozen modules (overwritten by direct HF load)")
        return orig_fn(model, *args, **kwargs)

    return parallelize_with_frozen_bf16


# --------------------------------------------------------------------------------------------
# Shared Pallas grouped-experts execution helper for dequantized BF16 expert weights.
# --------------------------------------------------------------------------------------------
def _run_pallas_grouped_experts(
    self,
    x_RD: torch.Tensor,
    num_tokens_per_expert_E: torch.Tensor,
    w1_EFD_bf16: torch.Tensor,
    w2_EDF_bf16: torch.Tensor,
    w3_EFD_bf16: torch.Tensor,
    row_scale_R: torch.Tensor | None = None,
) -> torch.Tensor:
    if not self.use_pallas_grouped_mm:
        offsets_E = torch.cumsum(num_tokens_per_expert_E, dim=0, dtype=torch.int32)
        h_RF = self._activation_fn(self._grouped_mm(A=x_RD.bfloat16(), weight_EOI=w1_EFD_bf16, offs=offsets_E))
        h_RF = h_RF * self._grouped_mm(A=x_RD.bfloat16(), weight_EOI=w3_EFD_bf16, offs=offsets_E)
        return self._grouped_mm(A=h_RF, weight_EOI=w2_EDF_bf16, offs=offsets_E).type_as(x_RD)

    w1_EKN = w1_EFD_bf16.transpose(-2, -1)
    w2_EKN = w2_EDF_bf16.transpose(-2, -1)
    w3_EKN = w3_EFD_bf16.transpose(-2, -1)
    if self._use_transposed_w2_rhs:
        w2_EKN = w2_EKN.transpose(-2, -1)

    x_bf16 = x_RD.bfloat16()
    tile_m = self.pallas_grouped_mm_tile_m
    num_rows = x_bf16.shape[0]
    pad_rows = -num_rows % tile_m
    if pad_rows:
        x_bf16 = F.pad(x_bf16, (0, 0, 0, pad_rows))
        num_tokens_per_expert_E = torch.cat(
            [num_tokens_per_expert_E[:-1], num_tokens_per_expert_E[-1:] + pad_rows]
        )
    tile_k = self.pallas_grouped_mm_tile_k
    tile_n = self.pallas_grouped_mm_tile_n
    tgmm_tiling = self.pallas_grouped_mm_tgmm_tiling
    w2_tgmm_tiling = self.pallas_grouped_mm_w2_tgmm_tiling or tgmm_tiling
    if self._use_transposed_w2_rhs:
        w2_tgmm_tiling = self.pallas_grouped_mm_w2_transposed_tgmm_tiling or w2_tgmm_tiling

    if self.use_pallas_paired_grouped_mm:
        gate_RF, up_RF = paired_segment_matmul_pallas_2d(
            x_bf16,
            w1_EKN,
            w3_EKN,
            num_tokens_per_expert_E,
            tile_m=tile_m,
            tgmm_tile_m=tgmm_tiling[0],
            tgmm_tile_k=tgmm_tiling[1],
        )
    else:
        gate_RF = segment_matmul_pallas_2d(
            x_bf16, w1_EKN, num_tokens_per_expert_E, tile_m=tile_m, tile_k=tile_k, tile_n=tile_n, tgmm_tiling=tgmm_tiling
        )
        up_RF = segment_matmul_pallas_2d(
            x_bf16, w3_EKN, num_tokens_per_expert_E, tile_m=tile_m, tile_k=tile_k, tile_n=tile_n, tgmm_tiling=tgmm_tiling
        )
    hidden_RF = self._activation_fn(gate_RF) * up_RF
    if row_scale_R is not None:
        if pad_rows:
            row_scale_R = F.pad(row_scale_R, (0, pad_rows))
        hidden_RF = (hidden_RF.float() * row_scale_R.reshape(-1, 1)).to(hidden_RF.dtype)
    return segment_matmul_pallas_2d(
        hidden_RF,
        w2_EKN,
        num_tokens_per_expert_E,
        tile_m=tile_m,
        tile_k=tile_k,
        tile_n=tile_n,
        tgmm_tiling=w2_tgmm_tiling,
        transpose_rhs=self._use_transposed_w2_rhs,
    )[:num_rows].type_as(x_RD)


# --------------------------------------------------------------------------------------------
# INT8 QLoRA: quantizes BOTH frozen nn.Linear AND frozen TpuGroupedExperts (39.48B params total).
# --------------------------------------------------------------------------------------------
_INT8_CLS_CACHE: dict[type, type] = {}
_INT8_EXP_CLS_CACHE: dict[type, type] = {}


def _int8_linear_forward(self, x: torch.Tensor) -> torch.Tensor:
    w_q = self.weight
    y = F.linear(x, w_q.to(x.dtype)) * self.weight_scale.to(x.dtype)
    if self.bias is not None:
        y = y + self.bias
    lora_a = getattr(self, "lora_a", None)
    if lora_a is not None:
        y = y + self._lora_scaling * self.lora_b(lora_a(x))
    return y


def _int8_experts_forward(
    self,
    x_RD: torch.Tensor,
    num_tokens_per_expert_E: torch.Tensor,
    row_scale_R: torch.Tensor | None = None,
) -> torch.Tensor:
    from torch.distributed.tensor import DTensor

    w1_q = self.w1_EFD.to_local() if isinstance(self.w1_EFD, DTensor) else self.w1_EFD
    w2_q = self.w2_EDF.to_local() if isinstance(self.w2_EDF, DTensor) else self.w2_EDF
    w3_q = self.w3_EFD.to_local() if isinstance(self.w3_EFD, DTensor) else self.w3_EFD
    w1_bf16 = w1_q.bfloat16() * self.w1_scale.unsqueeze(-1)
    w2_bf16 = w2_q.bfloat16() * self.w2_scale.unsqueeze(-1)
    w3_bf16 = w3_q.bfloat16() * self.w3_scale.unsqueeze(-1)
    return _run_pallas_grouped_experts(self, x_RD, num_tokens_per_expert_E, w1_bf16, w2_bf16, w3_bf16, row_scale_R)


def _int8_cls(cls: type) -> type:
    if cls not in _INT8_CLS_CACHE:
        _INT8_CLS_CACHE[cls] = type(f"Int8{cls.__name__}", (cls,), {"forward": _int8_linear_forward})
    return _INT8_CLS_CACHE[cls]


def _int8_exp_cls(cls: type) -> type:
    if cls not in _INT8_EXP_CLS_CACHE:
        _INT8_EXP_CLS_CACHE[cls] = type(f"Int8{cls.__name__}", (cls,), {"forward": _int8_experts_forward})
    return _INT8_EXP_CLS_CACHE[cls]


@torch.no_grad()
def quantize_flint_int8_(model_parts: list[nn.Module], gather_dtype: str = "int8") -> dict[str, Any]:
    from torch.distributed.tensor import DTensor, Shard

    n_lin, n_exp_mod, lin_params, exp_params, max_rel_err = 0, 0, 0, 0, 0.0
    for part in model_parts:
        for fqn, mod in part.named_modules():
            if not fqn.startswith("layers."):
                continue
            if isinstance(mod, nn.Linear):
                p = mod._parameters.get("weight")
                if p is None or p.requires_grad or not p.is_floating_point():
                    continue
                is_dt = isinstance(p, DTensor)
                local = p.to_local() if is_dt else p.data
                w = local.float()
                shard_dims = [pl.dim for pl in p.placements if isinstance(pl, Shard)] if is_dt else []
                amax = w.abs().amax(dim=1)
                if shard_dims and shard_dims[0] == 1:
                    torch.distributed.all_reduce(amax, op=torch.distributed.ReduceOp.MAX, group=p.device_mesh.get_group())
                scale = (amax.clamp(min=1e-12) / 127.0).to(torch.bfloat16)
                q = torch.round(w / scale.float()[:, None]).clamp_(-127, 127).to(torch.int8)
                if n_lin < 4:
                    err = ((q.float() * scale.float()[:, None] - w).norm() / w.norm().clamp(min=1e-12)).item()
                    max_rel_err = max(max_rel_err, err)
                if is_dt and shard_dims and shard_dims[0] == 0:
                    scale_full = DTensor.from_local(
                        scale, p.device_mesh, [Shard(0)], run_check=False, shape=(p.shape[0],), stride=(1,)
                    ).full_tensor()
                else:
                    scale_full = scale
                new_p = (
                    DTensor.from_local(q, p.device_mesh, p.placements, run_check=False, shape=p.shape, stride=p.stride())
                    if is_dt
                    else q
                )
                mod._parameters["weight"] = nn.Parameter(new_p, requires_grad=False)
                mod.register_buffer("weight_scale", scale_full.contiguous(), persistent=False)
                if gather_dtype == "int8":
                    for rc in getattr(mod, "_simple_fsdp_parametrization_list", []):
                        rc.param_dtype = None
                mod.__class__ = _int8_cls(type(mod))
                n_lin += 1
                lin_params += p.numel()
            elif isinstance(mod, TpuGroupedExperts):
                assert not mod.use_kernel_native_weight_layout, "INT8 QLoRA expects canonical [E, out, in] layout"
                for pname, sname in (("w1_EFD", "w1_scale"), ("w2_EDF", "w2_scale"), ("w3_EFD", "w3_scale")):
                    p = mod._parameters.get(pname)
                    if p is None or p.requires_grad or not p.is_floating_point():
                        continue
                    is_dt = isinstance(p, DTensor)
                    local = p.to_local() if is_dt else p.data
                    w = local.float()  # [E_loc, out_loc, in_f]
                    amax = w.abs().amax(dim=-1)  # [E_loc, out_loc]
                    scale = (amax.clamp(min=1e-12) / 127.0).to(torch.bfloat16)
                    q = torch.round(w / scale.float().unsqueeze(-1)).clamp_(-127, 127).to(torch.int8)
                    if n_exp_mod < 2:
                        err = ((q.float() * scale.float().unsqueeze(-1) - w).norm() / w.norm().clamp(min=1e-12)).item()
                        max_rel_err = max(max_rel_err, err)
                    if is_dt:
                        scale_full = DTensor.from_local(
                            scale.contiguous(),
                            p.device_mesh,
                            p.placements,
                            run_check=False,
                            shape=(p.shape[0], p.shape[1]),
                            stride=(p.shape[1], 1),
                        ).full_tensor()
                        new_p = DTensor.from_local(
                            q.contiguous(), p.device_mesh, p.placements, run_check=False, shape=p.shape, stride=p.stride()
                        )
                    else:
                        scale_full = scale
                        new_p = q
                    mod._parameters[pname] = nn.Parameter(new_p, requires_grad=False)
                    mod.register_buffer(sname, scale_full.contiguous(), persistent=False)
                    exp_params += p.numel()
                if gather_dtype == "int8":
                    for rc in getattr(mod, "_simple_fsdp_parametrization_list", []):
                        rc.param_dtype = None
                mod.__class__ = _int8_exp_cls(type(mod))
                n_exp_mod += 1
    gc.collect()
    return {
        "int8_linear_modules": n_lin,
        "int8_expert_modules": n_exp_mod,
        "int8_linear_params": int(lin_params),
        "int8_expert_params": int(exp_params),
        "int8_total_quantized_params": int(lin_params + exp_params),
        "int8_rel_fro_err_sample": round(max_rel_err, 6),
        "int8_gather_dtype": gather_dtype,
    }


# --------------------------------------------------------------------------------------------
# NF4 QLoRA: 4-bit NormalFloat (block-64) + 8-bit double-quantized absmax (4.125 bits/param).
# Quantizes BOTH frozen nn.Linear AND frozen TpuGroupedExperts (39.48B params total).
# --------------------------------------------------------------------------------------------
NF4_CODE = [
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
]
NF4_BLOCK = 64
_NF4_CLS_CACHE: dict[type, type] = {}
_NF4_EXP_CLS_CACHE: dict[type, type] = {}


def _nf4_decode(idx: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    i = idx.to(torch.int32)
    bits = [(i & (1 << k)) != 0 for k in range(4)]
    level = [torch.where(bits[0], NF4_CODE[2 * j + 1], NF4_CODE[2 * j]) for j in range(8)]
    level = [torch.where(bits[1], level[2 * j + 1], level[2 * j]) for j in range(4)]
    level = [torch.where(bits[2], level[2 * j + 1], level[2 * j]) for j in range(2)]
    return torch.where(bits[3], level[1], level[0]).to(dtype)


def _nf4_linear_forward(self, x: torch.Tensor) -> torch.Tensor:
    st = self.weight
    out_f, in_f = self.out_features, self.in_features
    nb = in_f // NF4_BLOCK
    packed, acode = st[:, : in_f // 2], st[:, in_f // 2 :]
    idx = torch.stack([packed & 0x0F, packed >> 4], dim=-1).reshape(out_f, in_f)
    absmax = (acode.to(torch.float32) * (1.0 / 255.0)) * self.weight_rowmax.to(torch.float32)[:, None]
    w = (_nf4_decode(idx, torch.bfloat16).reshape(out_f, nb, NF4_BLOCK) * absmax.to(torch.bfloat16)[..., None]).reshape(
        out_f, in_f
    )
    y = F.linear(x, w.to(x.dtype))
    if self.bias is not None:
        y = y + self.bias
    lora_a = getattr(self, "lora_a", None)
    if lora_a is not None:
        y = y + self._lora_scaling * self.lora_b(lora_a(x))
    return y


def _nf4_decode_3d(st: torch.Tensor, rowmax: torch.Tensor, in_f: int) -> torch.Tensor:
    e_dim, out_f, _ = st.shape
    nb = in_f // NF4_BLOCK
    packed, acode = st[..., : in_f // 2], st[..., in_f // 2 :]
    idx = torch.stack([packed & 0x0F, packed >> 4], dim=-1).reshape(e_dim, out_f, in_f)
    absmax = (acode.to(torch.float32) * (1.0 / 255.0)).to(torch.bfloat16) * rowmax.unsqueeze(-1)
    return (_nf4_decode(idx, torch.bfloat16).reshape(e_dim, out_f, nb, NF4_BLOCK) * absmax.unsqueeze(-1)).reshape(
        e_dim, out_f, in_f
    )


def _nf4_experts_forward(
    self,
    x_RD: torch.Tensor,
    num_tokens_per_expert_E: torch.Tensor,
    row_scale_R: torch.Tensor | None = None,
) -> torch.Tensor:
    from torch.distributed.tensor import DTensor

    s1 = self.w1_EFD.to_local() if isinstance(self.w1_EFD, DTensor) else self.w1_EFD
    s2 = self.w2_EDF.to_local() if isinstance(self.w2_EDF, DTensor) else self.w2_EDF
    s3 = self.w3_EFD.to_local() if isinstance(self.w3_EFD, DTensor) else self.w3_EFD
    w1_bf16 = _nf4_decode_3d(s1, self.w1_rowmax, self._nf4_w1_in)
    w2_bf16 = _nf4_decode_3d(s2, self.w2_rowmax, self._nf4_w2_in)
    w3_bf16 = _nf4_decode_3d(s3, self.w3_rowmax, self._nf4_w3_in)
    return _run_pallas_grouped_experts(self, x_RD, num_tokens_per_expert_E, w1_bf16, w2_bf16, w3_bf16, row_scale_R)


def _nf4_cls(cls: type) -> type:
    if cls not in _NF4_CLS_CACHE:
        _NF4_CLS_CACHE[cls] = type(f"NF4{cls.__name__}", (cls,), {"forward": _nf4_linear_forward})
    return _NF4_CLS_CACHE[cls]


def _nf4_exp_cls(cls: type) -> type:
    if cls not in _NF4_EXP_CLS_CACHE:
        _NF4_EXP_CLS_CACHE[cls] = type(f"NF4{cls.__name__}", (cls,), {"forward": _nf4_experts_forward})
    return _NF4_EXP_CLS_CACHE[cls]


def _pack_nf4_lastdim(w_fp32: torch.Tensor, mids: list[float]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    lead = w_fp32.shape[:-1]
    in_f = w_fp32.shape[-1]
    nb = in_f // NF4_BLOCK
    blocks = w_fp32.reshape(*lead, nb, NF4_BLOCK)
    absmax = blocks.abs().amax(-1)
    rowmax = absmax.amax(-1).clamp(min=1e-12).to(torch.bfloat16)
    acode = torch.round(absmax / rowmax.float().unsqueeze(-1) * 255.0).clamp_(1, 255)
    absmax_dq = acode / 255.0 * rowmax.float().unsqueeze(-1)
    normed = (blocks / absmax_dq.unsqueeze(-1)).clamp_(-1.0, 1.0)
    idx = torch.zeros(normed.shape, dtype=torch.int32, device=normed.device)
    for m in mids:
        idx += (normed > m).to(torch.int32)
    idx_flat = idx.reshape(*lead, in_f)
    packed = (idx_flat[..., 0::2] | (idx_flat[..., 1::2] << 4)).to(torch.uint8)
    store = torch.cat([packed, acode.to(torch.uint8)], dim=-1).contiguous()
    return store, rowmax, absmax_dq


@torch.no_grad()
def quantize_flint_nf4_(model_parts: list[nn.Module]) -> dict[str, Any]:
    from torch.distributed.tensor import DTensor, Shard

    mids = [(NF4_CODE[j] + NF4_CODE[j + 1]) / 2.0 for j in range(15)]
    n_lin, n_exp_mod, lin_params, exp_params, max_rel_err = 0, 0, 0, 0, 0.0
    fallback: list[str] = []

    for part in model_parts:
        for fqn, mod in part.named_modules():
            if not fqn.startswith("layers."):
                continue
            if isinstance(mod, nn.Linear):
                p = mod._parameters.get("weight")
                if p is None or p.requires_grad or not p.is_floating_point():
                    continue
                is_dt = isinstance(p, DTensor)
                shard_dims = [pl.dim for pl in p.placements if isinstance(pl, Shard)] if is_dt else []
                if (shard_dims and shard_dims[0] != 0) or (p.shape[1] % NF4_BLOCK != 0):
                    fallback.append(fqn)
                    continue
                local = p.to_local() if is_dt else p.data
                w = local.float()
                store, rowmax, absmax_dq = _pack_nf4_lastdim(w, mids)
                if n_lin < 4:
                    idx = torch.stack([store[:, : w.shape[1] // 2] & 0x0F, store[:, : w.shape[1] // 2] >> 4], dim=-1).reshape_as(w)
                    deq = (_nf4_decode(idx, torch.float32).reshape(w.shape[0], -1, NF4_BLOCK) * absmax_dq.unsqueeze(-1)).reshape_as(w)
                    max_rel_err = max(max_rel_err, ((deq - w).norm() / w.norm().clamp(min=1e-12)).item())
                if is_dt and shard_dims:
                    rowmax_full = DTensor.from_local(
                        rowmax, p.device_mesh, [Shard(0)], run_check=False, shape=(p.shape[0],), stride=(1,)
                    ).full_tensor()
                else:
                    rowmax_full = rowmax
                if is_dt:
                    gshape = (p.shape[0], store.shape[1])
                    new_p = DTensor.from_local(store, p.device_mesh, p.placements, run_check=False, shape=gshape, stride=(store.shape[1], 1))
                else:
                    new_p = store
                mod._parameters["weight"] = nn.Parameter(new_p, requires_grad=False)
                mod.register_buffer("weight_rowmax", rowmax_full.contiguous(), persistent=False)
                for rc in getattr(mod, "_simple_fsdp_parametrization_list", []):
                    rc.param_dtype = None
                mod.__class__ = _nf4_cls(type(mod))
                n_lin += 1
                lin_params += p.numel()
            elif isinstance(mod, TpuGroupedExperts):
                assert not mod.use_kernel_native_weight_layout
                ok_all = all(
                    (mod._parameters[pn].shape[-1] % NF4_BLOCK == 0)
                    for pn in ("w1_EFD", "w2_EDF", "w3_EFD")
                    if pn in mod._parameters
                )
                if not ok_all:
                    fallback.append(fqn)
                    continue
                for pname, rname, in_attr in (
                    ("w1_EFD", "w1_rowmax", "_nf4_w1_in"),
                    ("w2_EDF", "w2_rowmax", "_nf4_w2_in"),
                    ("w3_EFD", "w3_rowmax", "_nf4_w3_in"),
                ):
                    p = mod._parameters[pname]
                    in_f = int(p.shape[-1])
                    setattr(mod, in_attr, in_f)
                    is_dt = isinstance(p, DTensor)
                    local = p.to_local() if is_dt else p.data
                    w = local.float()
                    store, rowmax, _ = _pack_nf4_lastdim(w, mids)
                    if is_dt:
                        rowmax_full = DTensor.from_local(
                            rowmax.contiguous(),
                            p.device_mesh,
                            p.placements,
                            run_check=False,
                            shape=(p.shape[0], p.shape[1]),
                            stride=(p.shape[1], 1),
                        ).full_tensor()
                        gshape = (p.shape[0], p.shape[1], store.shape[2])
                        gstride = (p.shape[1] * store.shape[2], store.shape[2], 1)
                        new_p = DTensor.from_local(store, p.device_mesh, p.placements, run_check=False, shape=gshape, stride=gstride)
                    else:
                        rowmax_full = rowmax
                        new_p = store
                    mod._parameters[pname] = nn.Parameter(new_p, requires_grad=False)
                    mod.register_buffer(rname, rowmax_full.contiguous(), persistent=False)
                    exp_params += p.numel()
                for rc in getattr(mod, "_simple_fsdp_parametrization_list", []):
                    rc.param_dtype = None
                mod.__class__ = _nf4_exp_cls(type(mod))
                n_exp_mod += 1
    gc.collect()
    return {
        "nf4_linear_modules": n_lin,
        "nf4_expert_modules": n_exp_mod,
        "nf4_linear_params": int(lin_params),
        "nf4_expert_params": int(exp_params),
        "nf4_total_quantized_params": int(lin_params + exp_params),
        "nf4_block": NF4_BLOCK,
        "nf4_rel_fro_err_sample": round(max_rel_err, 6),
        "nf4_fallback_modules": fallback[:10],
        "bits_per_quantized_param": 4.0 + 8.0 / NF4_BLOCK,
    }


# --------------------------------------------------------------------------------------------
# Direct Sharded Flint HF Safetensors Loader (reads only local DTensor slices + 16 local experts/layer)
# --------------------------------------------------------------------------------------------
_EXPERT_PROJ_MAP = {"w1_EFD": "gate_proj", "w2_EDF": "down_proj", "w3_EFD": "up_proj"}

_LAYER_2D_MAP = {
    "attention_norm.weight": "input_layernorm.weight",
    "ffn_norm.weight": "post_attention_layernorm.weight",
    "attention.sinks": "self_attn.softmax_offset",
    "attention.q_norm.weight": "self_attn.q_norm.weight",
    "attention.k_norm.weight": "self_attn.k_norm.weight",
    "attention.qkv_linear.wq.weight": "self_attn.q_proj.weight",
    "attention.qkv_linear.wk.weight": "self_attn.k_proj.weight",
    "attention.qkv_linear.wv.weight": "self_attn.v_proj.weight",
    "attention.wo.weight": "self_attn.o_proj.weight",
    "feed_forward.w1.weight": "mlp.gate_proj.weight",
    "feed_forward.w2.weight": "mlp.down_proj.weight",
    "feed_forward.w3.weight": "mlp.up_proj.weight",
    "moe.router.gate.weight": "mlp.gate.weight",
    "moe.shared_experts.w1.weight": "mlp.shared_expert.gate_proj.weight",
    "moe.shared_experts.w2.weight": "mlp.shared_expert.down_proj.weight",
    "moe.shared_experts.w3.weight": "mlp.shared_expert.up_proj.weight",
}

_TOP_LEVEL_MAP = {
    "tok_embeddings.weight": "model.embed_tokens.weight",
    "norm.weight": "model.norm.weight",
    "lm_head.weight": "lm_head.weight",
    "output.weight": "lm_head.weight",
}


@torch.no_grad()
def direct_sharded_flint_hf_load_(model_wrapper, path: str) -> dict[str, Any]:
    from safetensors import safe_open
    from torch.distributed.tensor import DTensor
    from torch.distributed.tensor._utils import compute_local_shape_and_global_offset

    t0 = time.time()
    with open(os.path.join(path, "model.safetensors.index.json")) as f:
        weight_map: dict[str, str] = json.load(f)["weight_map"]
    handles: dict[str, Any] = {}

    def get_slice(hf_key: str):
        fname = weight_map[hf_key]
        if fname not in handles:
            handles[fname] = safe_open(os.path.join(path, fname), framework="pt")
        return handles[fname].get_slice(hf_key)

    n_loaded, n_bytes = 0, 0
    skipped: list[str] = []

    for part in model_wrapper.model:
        for raw_key, val in part.state_dict().items():
            key = raw_key.replace("_checkpoint_wrapped_module.", "")
            if ".lora_a." in key or ".lora_b." in key:
                continue
            if key.endswith(("expert_bias_E", "freqs_cis", "inv_freq", "cache")):
                continue

            if isinstance(val, DTensor):
                lshape, goff = compute_local_shape_and_global_offset(val.shape, val.device_mesh, val.placements)
                local = val._local_tensor
            else:
                lshape, goff, local = tuple(val.shape), (0,) * val.dim(), val

            m_exp = re.match(r"layers\.(\d+)\.moe\.routed_experts\.inner_experts\.(w1_EFD|w2_EDF|w3_EFD)$", key)
            if m_exp is not None:
                layer_idx, wname = m_exp.group(1), m_exp.group(2)
                proj = _EXPERT_PROJ_MAP[wname]
                e_off, r_off, c_off = goff
                e_len, r_len, c_len = lshape
                buf_cpu = torch.empty((e_len, r_len, c_len), dtype=torch.bfloat16)
                for e_i in range(e_len):
                    e_global = e_off + e_i
                    hf_k = f"model.layers.{layer_idx}.mlp.experts.{e_global}.{proj}.weight"
                    sl = get_slice(hf_k)
                    t = sl[r_off : r_off + r_len, c_off : c_off + c_len]
                    buf_cpu[e_i].copy_(t)
                    n_bytes += t.numel() * t.element_size()
                local.copy_(buf_cpu.to(dtype=local.dtype, device=local.device))
                n_loaded += 1
                continue

            hf_key = _TOP_LEVEL_MAP.get(key)
            if hf_key is None:
                m_l = re.match(r"layers\.(\d+)\.(.+)$", key)
                if m_l and m_l.group(2) in _LAYER_2D_MAP:
                    hf_key = f"model.layers.{m_l.group(1)}.{_LAYER_2D_MAP[m_l.group(2)]}"

            if hf_key is None or hf_key not in weight_map:
                skipped.append(key)
                continue

            sl = get_slice(hf_key)
            idx = tuple(slice(goff[d], goff[d] + lshape[d]) for d in range(len(lshape)))
            t = sl[idx] if len(idx) else sl[:]
            if tuple(t.shape) != tuple(local.shape):
                raise RuntimeError(f"Shape mismatch for {key} <- {hf_key}: {tuple(t.shape)} vs {tuple(local.shape)}")
            local.copy_(t.to(dtype=local.dtype, device=local.device))
            n_loaded += 1
            n_bytes += t.numel() * t.element_size()

    try:
        from torch_tpu._internal.sync import synchronize

        synchronize(wait=True)
    except Exception:
        pass
    model_wrapper.cached_state_dict = model_wrapper._get_state_dict()
    info = {
        "loader": "direct_sharded_flint_safetensors",
        "source_path": path,
        "tensors_loaded": n_loaded,
        "bytes_read_this_rank": int(n_bytes),
        "seconds": round(time.time() - t0, 2),
        "skipped_keys": skipped[:20],
    }
    _log(f"direct Flint HF load: {info}")
    if skipped:
        raise RuntimeError(f"Unmapped Flint parameters (would train on random weights!): {skipped[:10]}")
    return info


# --------------------------------------------------------------------------------------------
# PEFT Adapter Export (saves both full adapter AND universal attention-only adapter)
# --------------------------------------------------------------------------------------------
def _full_tensor(p: torch.Tensor) -> torch.Tensor:
    from torch.distributed.tensor import DTensor

    return p.full_tensor() if isinstance(p, DTensor) else p


@torch.no_grad()
def export_flint_peft_adapters(model_parts: list[nn.Module], root_dir: str) -> dict[str, Any]:
    full_tensors: dict[str, torch.Tensor] = {}
    attn_tensors: dict[str, torch.Tensor] = {}
    full_targets: set[str] = set()
    attn_targets: set[str] = set()
    n_full, n_attn = 0, 0

    for part in model_parts:
        for fqn, mod in part.named_modules():
            if not (hasattr(mod, "lora_a") and hasattr(mod, "lora_b")):
                continue
            clean = fqn.replace("_checkpoint_wrapped_module.", "")
            m = re.match(r"layers\.(\d+)\.(.+)$", clean)
            if m is None or m.group(2) not in TT2HF_FLINT:
                raise RuntimeError(f"Cannot map Flint LoRA module {clean} to HF name")
            layer_i, tt_sub = m.group(1), m.group(2)
            hf_sub = TT2HF_FLINT[tt_sub]
            leaf = hf_sub.rsplit(".", 1)[-1]
            prefix = f"{_RUN['hf_prefix']}.{layer_i}.{hf_sub}"
            a = _full_tensor(mod.lora_a._parameters["weight"]).to(torch.bfloat16)
            b = _full_tensor(mod.lora_b._parameters["weight"]).to(torch.bfloat16)
            full_targets.add(leaf)
            n_full += 1
            is_attn = hf_sub.startswith("self_attn.")
            if is_attn:
                attn_targets.add(leaf)
                n_attn += 1
            if _rank() == 0:
                ca, cb = a.cpu().contiguous(), b.cpu().contiguous()
                full_tensors[f"{prefix}.lora_A.weight"] = ca
                full_tensors[f"{prefix}.lora_B.weight"] = cb
                if is_attn:
                    attn_tensors[f"{prefix}.lora_A.weight"] = ca
                    attn_tensors[f"{prefix}.lora_B.weight"] = cb

    info: dict[str, Any] = {
        "full_lora_modules": n_full,
        "attn_lora_modules": n_attn,
        "full_hf_target_modules": sorted(full_targets),
        "attn_hf_target_modules": sorted(attn_targets),
    }
    if _rank() == 0:
        from safetensors.torch import save_file

        def _write_bundle(sub_dir: str, tdict: dict[str, torch.Tensor], tmods: set[str]) -> dict[str, Any]:
            os.makedirs(sub_dir, exist_ok=True)
            save_file(tdict, os.path.join(sub_dir, "adapter_model.safetensors"), metadata={"format": "pt"})
            cfg = {
                "peft_type": "LORA",
                "task_type": "CAUSAL_LM",
                "base_model_name_or_path": _RUN["base_model"],
                "r": int(_RUN["rank"]),
                "lora_alpha": float(_RUN["alpha"]),
                "lora_dropout": 0.0,
                "bias": "none",
                "fan_in_fan_out": False,
                "inference_mode": True,
                "init_lora_weights": True,
                "target_modules": sorted(tmods),
                "modules_to_save": None,
                "use_rslora": False,
                "use_dora": False,
                "rank_pattern": {},
                "alpha_pattern": {},
                "layers_to_transform": None,
                "layers_pattern": None,
                "revision": None,
            }
            with open(os.path.join(sub_dir, "adapter_config.json"), "w") as f:
                json.dump(cfg, f, indent=2)
            nbytes = sum(t.numel() * t.element_size() for t in tdict.values())
            return {"dir": sub_dir, "tensors": len(tdict), "bytes": int(nbytes)}

        full_info = _write_bundle(os.path.join(root_dir, "peft_adapter"), full_tensors, full_targets)
        attn_info = _write_bundle(os.path.join(root_dir, "peft_adapter_attn"), attn_tensors, attn_targets)
        info["full_bundle"] = full_info
        info["attn_bundle"] = attn_info

        # Also mirror real 40B adapters to canonical /models/adapters/ for serving benchmarks
        if _RUN.get("is_real_40b"):
            canon_base = os.path.join("/models/adapters", os.environ.get("LORA_FT_ADAPTER_ALIAS", _RUN["config"]))
            shutil.copytree(attn_info["dir"], canon_base, dirs_exist_ok=True)
            shutil.copytree(full_info["dir"], canon_base + "-full", dirs_exist_ok=True)
            info["canonical_serving_adapter"] = canon_base
            info["canonical_full_adapter"] = canon_base + "-full"
    return info


# --------------------------------------------------------------------------------------------
# Trainer & Checkpointer Hooks
# --------------------------------------------------------------------------------------------
def _install_patches() -> None:
    from torchtitan import trainer as tt_trainer
    from torchtitan.components.checkpointer import base as ckpt_base

    if getattr(ckpt_base.BaseCheckpointManager, "_flint_lora_patched", False):
        return

    orig_load = ckpt_base.BaseCheckpointManager.load

    def load_then_quantize(self, step: int = -1) -> bool:
        t0 = time.time()
        if self.enable and self.initial_load_path and self.initial_load_in_hf:
            info = direct_sharded_flint_hf_load_(self.states[ckpt_base.MODEL], self.initial_load_path)
            _RUN["stats"]["hf_load"] = info
            ok = True
        else:
            ok = orig_load(self, step)
        _RUN["stats"]["checkpoint_load_s"] = round(time.time() - t0, 2)
        _RUN["stats"]["mem_after_load"] = _device_mem_gib()

        if _RUN["quant"] in ("int8", "nf4") and self.enable:
            t1 = time.time()
            wrapper = self.states[ckpt_base.MODEL]
            if _RUN["quant"] == "int8":
                qinfo = quantize_flint_int8_(wrapper.model, os.environ.get("LORA_FT_INT8_GATHER", "int8"))
            else:
                qinfo = quantize_flint_nf4_(wrapper.model)
            wrapper.cached_state_dict = wrapper._get_state_dict()
            gc.collect()
            try:
                from torch_tpu._internal.sync import synchronize

                synchronize(wait=True)
            except Exception:
                pass
            qinfo["quantize_s"] = round(time.time() - t1, 2)
            _RUN["stats"].update(qinfo)
            _RUN["stats"]["mem_after_quant"] = _device_mem_gib()
            _log(f"{_RUN['quant']} quantized Flint base: {qinfo}")
        _log(
            f"checkpoint ready in {_RUN['stats']['checkpoint_load_s']}s; "
            f"mem={_RUN['stats'].get('mem_after_quant', _RUN['stats']['mem_after_load'])}"
        )
        return ok

    ckpt_base.BaseCheckpointManager.load = load_then_quantize
    ckpt_base.BaseCheckpointManager._flint_lora_patched = True

    orig_train = tt_trainer.Trainer.train

    def train_then_export(self):
        t0 = time.time()
        out = orig_train(self)
        _RUN["stats"]["train_wall_s"] = round(time.time() - t0, 2)
        _RUN["stats"]["mem_end"] = _device_mem_gib()
        run = os.environ.get("LORA_FT_RUN_NAME", _RUN["config"])
        ts = os.environ.get("LORA_FT_TS", _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
        root = os.path.join(os.environ.get("LORA_FT_OUT", "/models/flint-lora-ft/runs"), f"{run}_{ts}")
        try:
            t1 = time.time()
            einfo = export_flint_peft_adapters(self.model_parts, root)
            einfo["export_s"] = round(time.time() - t1, 2)
            _RUN["stats"]["export"] = einfo
        except Exception as e:
            _RUN["stats"]["export"] = {"error": f"{type(e).__name__}: {e}"}
        if _rank() == 0:
            os.makedirs(root, exist_ok=True)
            rec = {
                "config": _RUN["config"],
                "model": "RecursiveSI/flint",
                "quant": _RUN["quant"] or "bf16",
                "lora_rank": _RUN["rank"],
                "lora_alpha": _RUN["alpha"],
                "lora_targets_tt": _RUN["targets"],
                "world_size": int(os.environ.get("WORLD_SIZE", "1")),
                **_RUN["stats"],
            }
            with open(os.path.join(root, "run_stats.json"), "w") as f:
                json.dump(rec, f, indent=2)
            _log(f"run_stats -> {root}/run_stats.json : {json.dumps(rec)}")
        return out

    tt_trainer.Trainer.train = train_then_export


# --------------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------------
def chat_messages(sample: dict[str, Any]) -> list[dict[str, str]]:
    return sample["messages"]


def _is_not_none(sample: Any) -> bool:
    return sample is not None


def sft_dataloader(path: str | None = None, seed: int = 42) -> GrainDataLoader.Config:
    ds = SingleDatasetConfig(
        source=HuggingFaceRandomAccessSource.Config(
            path="json",
            split="train",
            load_dataset_kwargs={"data_files": path or os.environ.get("LORA_FT_DATA", DEFAULT_DATA)},
        ),
        processor=ChatProcessor.Config(messages_fn=chat_messages),
        post_filters=(_is_not_none,),
    )
    return GrainDataLoader.Config(dataset=ConcatThenSplitPackingConfig(dataset=ds), seed=seed, shuffle=True)


def _c4_test_loader():
    from torchtitan.experiments.tpu.datasets import c4_test_dataloader

    return c4_test_dataloader()


# --------------------------------------------------------------------------------------------
# Recipe Builder
# --------------------------------------------------------------------------------------------
def _flint_lora_spec(flavor: str, seq_len: int, rank: int, alpha: float, targets: list[str], use_pallas_kernels: bool = True):
    converters = [
        TpuTokenDispatcherConverter.Config(),
        TpuGroupedExpertsConverter.Config(),
        LoRAConverter.Config(rank=rank, alpha=alpha, target_modules=list(targets)),
    ]
    spec = flint_model_registry(
        flavor,
        max_context_length=seq_len,
        enable_sp=False,
        enable_ep=False,
        converters=converters,
    )
    if use_pallas_kernels:
        for _, attn, _, _ in spec.model.traverse(FlintSinkAttention.Config):
            attn.use_splash_kernel = True
        for _, exp, _, _ in spec.model.traverse(TpuGroupedExperts.Config):
            exp.use_pallas_grouped_mm = True
            exp.use_pallas_paired_grouped_mm = True
            tile_m = int(os.environ.get("FLINT_GMM_TILE_M", "256" if flavor == "40B" else "128"))
            exp.pallas_grouped_mm_tile_m = tile_m
            if flavor == "40B":
                exp.pallas_grouped_mm_tgmm_tiling = (tile_m, 2048, 0)
                exp.pallas_grouped_mm_w2_tgmm_tiling = (tile_m, 512, 0)
                exp.pallas_grouped_mm_w2_transposed_tgmm_tiling = (tile_m, 0, 2048)
            exp.use_kernel_native_weight_layout = False
    return dataclasses.replace(spec, parallelize_fn=_wrap_parallelize(spec.parallelize_fn))


def _base_flint_config(
    *,
    name: str,
    quant: str | None,
    flavor: str = "40B",
    seq_len: int = 4096,
    seqs_per_rank: int = 2,
    steps: int = 120,
    lr: float = 2e-4,
    rank: int = 16,
    alpha: float = 32.0,
    targets: list[str] | None = None,
    real_hf: bool = True,
) -> TPUTrainerConfig:
    tlist = list(targets or LORA_TARGETS_ALL)
    hf_path = _resolve_hf_path() if real_hf else None
    _RUN.update(
        {
            "config": name,
            "quant": quant,
            "rank": rank,
            "alpha": alpha,
            "targets": tlist,
            "is_real_40b": real_hf and flavor == "40B",
            "skip_frozen_init": real_hf and os.environ.get("LORA_FT_SKIP_FROZEN_INIT", "1") == "1",
        }
    )
    _RUN["stats"]["seq_len"] = seq_len
    _RUN["stats"]["seqs_per_rank"] = seqs_per_rank
    _install_patches()

    use_compile = os.environ.get("FLINT_LORA_COMPILE", "0") == "1"
    default_pallas_ce = "1" if seq_len >= 16384 else "0"
    use_pallas_ce = os.environ.get("FLINT_LORA_PALLAS_CE", default_pallas_ce) == "1" and flavor == "40B"
    _RUN["stats"]["pallas_fused_ce"] = use_pallas_ce

    cfg = TPUTrainerConfig(
        loss=PallasCrossEntropyLoss.Config() if use_pallas_ce else CrossEntropyLoss.Config(),
        hf_assets_path=hf_path or "./tests/assets/tokenizer",
        metrics=MetricsProcessor.Config(log_freq=1),
        model_spec=_flint_lora_spec(flavor, seq_len, rank, alpha, tlist, use_pallas_kernels=(flavor == "40B")),
        optimizer=default_adamw(lr=lr, weight_decay=0.0, implementation="foreach"),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=min(10, max(2, steps // 5)), decay_ratio=0.9, decay_type="cosine", min_lr_factor=0.1),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=seqs_per_rank * seq_len,
            max_context_length=seq_len,
            steps=steps,
            max_norm=1.0,
            dtype="float32",
            mixed_precision_param="bfloat16",
            mixed_precision_reduce="float32",
        ),
        dataloader=sft_dataloader() if real_hf else _c4_test_loader(),
        parallelism=ParallelismConfig(
            use_simple_fsdp=True,
            data_parallel_shard_degree=-1,
            fsdp_reshard_after_forward="always",
        ),
        comm=CommConfig(init_timeout_seconds=3600, train_timeout_seconds=3600),
        checkpoint=CheckpointManager.Config(
            enable=True,
            load_only=True,
            initial_load_path=hf_path,
            initial_load_in_hf=hf_path is not None,
            initial_load_model_only=hf_path is not None,
        ),
        activation_checkpoint=FullAC.Config(preserve_rng_state=False),
        compile=TpuTrainerCompileConfig(
            enable=use_compile,
            backend="tpu",
            components=["model"],
            compile_grad_norm=True,
        ),
        validator=Validator.Config(enable=False),
        tpu_config=TPUConfig(
            enable_amp=False,
            eager_mode=None if use_compile else "DEFER_AND_FUSE",
        ),
    )
    if use_pallas_ce:
        from torchtitan.config import derive
        from torchtitan.experiments.tpu.overrides.fused_loss import FusedLossLinearWrapper

        cfg.model_spec.model.lm_head = derive(cfg.model_spec.model.lm_head, FusedLossLinearWrapper.Config)
    cfg.debug.seed = 42
    return cfg


def flint_40b_lora_bf16() -> TPUTrainerConfig:
    return _base_flint_config(name="flint_40b_lora_bf16", quant=None)


def flint_40b_qlora_int8() -> TPUTrainerConfig:
    return _base_flint_config(name="flint_40b_qlora_int8", quant="int8")


def flint_40b_qlora_nf4() -> TPUTrainerConfig:
    return _base_flint_config(name="flint_40b_qlora_nf4", quant="nf4")


def flint_40b_lora_bf16_r64() -> TPUTrainerConfig:
    return _base_flint_config(name="flint_40b_lora_bf16_r64", quant=None, rank=64, alpha=128.0)


def flint_40b_lora_bf16_attn() -> TPUTrainerConfig:
    return _base_flint_config(name="flint_40b_lora_bf16_attn", quant=None, targets=LORA_TARGETS_ATTN)


def flint_40b_lora_bf16_16k() -> TPUTrainerConfig:
    return _base_flint_config(name="flint_40b_lora_bf16_16k", quant=None, seq_len=16384, seqs_per_rank=1, steps=40)


def flint_40b_qlora_int8_16k() -> TPUTrainerConfig:
    return _base_flint_config(name="flint_40b_qlora_int8_16k", quant="int8", seq_len=16384, seqs_per_rank=1, steps=40)


def flint_40b_lora_bf16_32k() -> TPUTrainerConfig:
    return _base_flint_config(name="flint_40b_lora_bf16_32k", quant=None, seq_len=32768, seqs_per_rank=1, steps=40)


def flint_40b_qlora_int8_32k() -> TPUTrainerConfig:
    return _base_flint_config(name="flint_40b_qlora_int8_32k", quant="int8", seq_len=32768, seqs_per_rank=1, steps=40)


def flint_40b_lora_bf16_64k() -> TPUTrainerConfig:
    return _base_flint_config(name="flint_40b_lora_bf16_64k", quant=None, seq_len=65536, seqs_per_rank=1, steps=30)


def flint_40b_qlora_int8_64k() -> TPUTrainerConfig:
    return _base_flint_config(name="flint_40b_qlora_int8_64k", quant="int8", seq_len=65536, seqs_per_rank=1, steps=30)


def flint_debug_lora_bf16() -> TPUTrainerConfig:
    return _base_flint_config(
        name="flint_debug_lora_bf16", quant=None, flavor="debugmodel", seq_len=512, seqs_per_rank=4, steps=6, real_hf=False
    )


def flint_debug_qlora_int8() -> TPUTrainerConfig:
    return _base_flint_config(
        name="flint_debug_qlora_int8", quant="int8", flavor="debugmodel", seq_len=512, seqs_per_rank=4, steps=6, real_hf=False
    )


def flint_debug_qlora_nf4() -> TPUTrainerConfig:
    return _base_flint_config(
        name="flint_debug_qlora_nf4", quant="nf4", flavor="debugmodel", seq_len=512, seqs_per_rank=4, steps=6, real_hf=False
    )
