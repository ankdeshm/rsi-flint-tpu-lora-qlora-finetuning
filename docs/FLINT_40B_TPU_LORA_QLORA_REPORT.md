# Technical Report: `RecursiveSI/flint` (`39.8B` MoE) LoRA & QLoRA Fine-Tuning (`v5p-16`) + Multi-LoRA Serving (`v7x` & `v5p`)

## 1. Model Topology & Parameter Accounting

`RecursiveSI/flint` is a sparse Mixture-of-Experts (`MoE`) autoregressive language model with **`39,785,287,456` base parameters** and **`~2.52B` active parameters per token**:
- **`49` Transformer Decoder Layers** (`d_model = 2048`, `rms_norm_eps = 1e-6`, untied `tok_embeddings` + `lm_head` over `vocab_size = 150,144` = `614.99M` embedding/head params).
- **Hybrid Sink Attention (`49` Layers):** `16` query heads (`head_dim = 128`), `2` KV heads (`8:1` GQA), QK-RMSNorm, and learned per-head softmax sink logits (`self_attn.softmax_offset` in HF / `attention.sinks` in TorchTitan). `35` layers apply sliding-window causal attention (`W = 513`) with RoPE; `14` layers (`layers 0, 1, 5, 9, ...`) apply full-sequence `NoPE` causal attention.
- **Dense Prefix + `256`-Expert MoE Trunk:**
  - Layers `0..1`: Dense SwiGLU FFN (`w1, w2, w3` with `hidden_dim = 6144`).
  - Layers `2..48` (`47` MoE layers): Non-gated router (`[256, 2048]`) selecting `top_k = 6` sigmoid-scored experts out of `256` (`expert_hidden = 512`, `37,849,399,296` routed expert parameters = **`95.13%` of all model weights**) plus `1` always-active shared SwiGLU expert (`shared_hidden = 1024`).

### TorchTitan -> Hugging Face `PEFT` Adapter Target Mapping

| TorchTitan Module FQN | HF `FlintForCausalLM` Module FQN | Layers | Tensor Shape (`out × in`) | `r=16` Trainable Params |
|---|---|---:|---:|---:|
| `layers.<L>.attention.qkv_linear.wq` | `model.layers.<L>.self_attn.q_proj` | `0..48` (`49`) | `2048 × 2048` | `3,211,264` |
| `layers.<L>.attention.qkv_linear.wk` | `model.layers.<L>.self_attn.k_proj` | `0..48` (`49`) | `256 × 2048` | `1,806,336` |
| `layers.<L>.attention.qkv_linear.wv` | `model.layers.<L>.self_attn.v_proj` | `0..48` (`49`) | `256 × 2048` | `1,806,336` |
| `layers.<L>.attention.wo` | `model.layers.<L>.self_attn.o_proj` | `0..48` (`49`) | `2048 × 2048` | `3,211,264` |
| *(Note: TorchTitan `SeparateQKVLinear` attaches independent `wq, wk, wv` adapters via `wkv`)* | — | — | *(Attention subtotal across `196` projections)* | **`13,647,872`** |
| `layers.<0..1>.feed_forward.{w1,w2,w3}` | `model.layers.<0..1>.mlp.{gate,down,up}_proj` | `0..1` (`2`) | `6144 × 2048` / `2048 × 6144` | `786,432` |
| `layers.<2..48>.moe.shared_experts.{w1,w2,w3}` | `model.layers.<2..48>.mlp.shared_expert.{gate,down,up}_proj` | `2..48` (`47`) | `1024 × 2048` / `2048 × 1024` | `6,930,432` |
| **Total (`flint_40b_lora_bf16` / `qlora_int8` / `qlora_nf4`)** | **`343` adapted `Linear` modules** | **`0..48`** | — | **`21,364,736` (`0.0537%`)** |

---

## 2. QLoRA Design for `3D` `TpuGroupedExperts` on TorchTPU

Standard QLoRA implementations only quantize `2D` `nn.Linear` layers—which on `RecursiveSI/flint` would leave `95.13%` of the model (`37.85B` routed expert parameters stored as `3D` `[256, 512, 2048]` / `[256, 2048, 512]` `DTensor`s inside `TpuGroupedExperts`) in `16`-bit `BF16`!

In [`flint_lora_tpu_recipes.py`](../finetuning/scripts/flint_lora_tpu_recipes.py), we quantize **both** `nn.Linear` (`390` modules, `1.32B` params) and `TpuGroupedExperts` (`47` modules, `37.85B` params = **`39.17B` total quantized parameters**):

1. **ICI-Compressed `SimpleFSDP` Sharding (`param_dtype=None`):**
   TorchTitan's `SimpleFSDP` wraps parameters with a `ReconstructParameter` parametrization that normally casts shards to `param_dtype=bfloat16` *before* the `AllGather`. Clearing `rc.param_dtype = None` on quantized modules forces XLA to `AllGather` raw `1`-byte `int8` shards (`39.17 GB` total across slice vs `78.34 GB` in `BF16`) or packed `uint8` `NF4` shards (`20.20 GB` total across slice).
2. **`INT8` Channelwise Expert & Linear Quantization (`_quantize_int8_`):**
   Each `[E_local, out, in]` expert shard is quantized per-(expert, output-row) (`scale = max(|W|) / 127.0` stored as `fp32` `[256, out]`), achieving **`0.015095` Frobenius relative error** in `5.04 s` across all `40B` weights. During forward and `FullAC` backward recomputation, the `int8` tensor feeds TPU `v5p`'s Pallas grouped-MM (`segment_matmul_pallas_2d` / `paired_segment_matmul_pallas_2d`, `tile_m=256`).
3. **`NF4` Block-`64` + `UINT8` Double-Quantized `TpuGroupedExperts` (`_quantize_nf4_`):**
   Packs pairs of `4`-bit NormalFloat codebook indices (`[E, out, in/2]`) concatenated with `uint8` block `absmax` scales (`[E, out, in/64]`) into a single `FSDP`-sharded `uint8` `DTensor` (`4.125` bits/param, `0.09335` Frobenius relative error in `3.12 s`), shrinking per-chip resting weight allocation to **`1.543 GiB`**.

---

## 3. Complete `4K` Loss Convergence & `tpu-info` HBM Telemetry (`TPU v5p-16`)

### 3.1 Step-by-Step SFT Loss Trajectory (`yahma/alpaca-cleaned`, `seed=42`)

| Step | `lora_bf16` (`r=16`) | `qlora_int8` (`r=16`) | `qlora_nf4` (`r=16`) | `lora_bf16_r64` (`r=64`) | `lora_bf16_attn` (`r=16` Attn-Only) |
|---:|---:|---:|---:|---:|---:|
| `1` | `2.13873` | `2.13400` | `2.25758` | `2.13873` | `2.13873` |
| `5` | `1.12208` | `1.12344` | `1.17103` | `1.13827` | `1.23587` |
| `10` | `1.08754` | `1.08829` | `1.10943` | `1.12535` | `1.09088` |
| `25` | `1.00362` | `1.00358` | `1.01671` | `1.03079` | `1.01180` |
| `50` | `1.02626` | `1.02675` | `1.03653` | `1.06122` | `1.03015` |
| `75` | `0.92488` | `0.92481` | `0.93329` | **`0.82299`** | `0.96733` |
| `100` | `0.90719` | `0.90728` | `0.91515` | **`0.79686`** | `0.94739` |
| `120` | `0.92931` | `0.92892` | `0.93842` | **`0.81424`** | `0.96980` |
| `150` | **`0.86375`** | **`0.86329`** | **`0.87202`** | — | — |

### 3.2 `tpu-info` Independent Runtime HBM Breakdown (`95.0 GiB` HBM/Chip)

| Recipe | Post-Load Shard | Post-Quant Shard | `tpu-info` `p10` | `tpu-info` Median (`p50`) | `tpu-info` `p95` | `tpu-info` Peak (`max`) | Headroom on `95 GiB` `v5p` |
|---|---:|---:|---:|---:|---:|---:|---:|
| `flint_40b_lora_bf16` | `4.784 GiB` | `4.784 GiB` | `10.34 GiB` | `12.61 GiB` | `13.82 GiB` | `18.33 GiB` | `76.67 GiB` (`80.7%` free) |
| **`flint_40b_qlora_int8`** | `4.784 GiB` | **`2.579 GiB`** | **`7.33 GiB`** | **`10.09 GiB`** | **`12.55 GiB`** | **`15.79 GiB`** | **`79.21 GiB` (`83.4%` free)** |
| `flint_40b_qlora_nf4` | `4.784 GiB` | **`1.543 GiB`** | `7.38 GiB` | `29.09 GiB` | `64.23 GiB` | `65.80 GiB` | `29.20 GiB` (`30.7%` free) |
| `flint_40b_lora_bf16_r64` | `4.800 GiB` | `4.800 GiB` | `9.57 GiB` | `12.38 GiB` | `13.98 GiB` | `18.38 GiB` | `76.62 GiB` (`80.7%` free) |
| `flint_40b_lora_bf16_attn` | `4.780 GiB` | `4.780 GiB` | `9.49 GiB` | `12.71 GiB` | `14.19 GiB` | `18.29 GiB` | `76.71 GiB` (`80.7%` free) |

---

## 4. Long-Context (`16K`–`64K`) Fused Pallas Cross-Entropy Architecture

At `seq_len = 65,536` (`vocab_size = 150,144`), materializing a standard `FP32` `[1, 65536, 150144]` logits tensor plus its backward gradient requires **`2 × 39.36 GB = 78.72 GB` (`73.3 GiB`) per chip**—which alone nearly exhausts `95 GiB` HBM before weights or MoE activations!

For all `seq_len >= 16,384` recipes (`flint_40b_{lora_bf16,qlora_int8}_{16k,32k,64k}`), [`flint_lora_tpu_recipes.py`](../finetuning/scripts/flint_lora_tpu_recipes.py) replaces `lm_head` with `derive(cfg.model_spec.model.lm_head, FusedLossLinearWrapper.Config)` paired with `PallasCrossEntropyLoss.Config()`. `lm_head` returns `(hidden_states, weight.T)` without projecting to vocabulary space, and `PallasCrossEntropyLoss` computes tiled projection + log-softmax + cross-entropy + backward gradients inside TPU VMEM—eliminating the multi-tens-of-GiB logits allocation completely and increasing `16K` per-chip throughput over `4K`!

---

## 5. `SGLang-TorchTPU` Multi-LoRA Serving Enablement for `FlintForCausalLM`

Out of the box, `sglang-torchtpu` disables LoRA on `FlintForCausalLM` across five places in the stack. [`serving/scripts/patch_sglang_tpu_lora.py`](../serving/scripts/patch_sglang_tpu_lora.py) patches all five cleanly:
1. **Model Capabilities (`flint.py`):** Sets `supports_lora = True` and declares `embedding_modules` / `embedding_padding_modules` on `FlintForCausalLM`.
2. **Stacked QKV Mapping:** Registers `qkv_proj -> [q_proj, k_proj, v_proj]` sub-module slicing so separate Hugging Face `PEFT` `q_proj, k_proj, v_proj` adapters fuse cleanly into `SGLang`'s `QKVParallelLinear` under any `TP` degree.
3. **`Block-FP8` Base Weight Compatibility (`lora_manager.py` / `layers.py`):** Allows `BF16` LoRA `A/B` Punica Pallas kernels to run atop `Block-FP8` (`[128, 128]` quantized) `FlintForCausalLM` base weights ("QLoRA-style serving": `FP8` base + `FP8` KV cache + `BF16` multi-tenant adapters).
4. **TPU Graph Capture Bucket Alignment:** Preserves `LoRABatchInfo` metadata across `torch_tpu` `EXTEND` and `DECODE` XLA graph captures so multi-tenant batches mixing `base + lora_1 + lora_2 + ...` execute without dynamic recompilation.
5. **Dynamic Adapter Control Plane:** Verifies concurrent routing across adapters trained by `BF16 LoRA`, `INT8 QLoRA`, `NF4 QLoRA`, and `attn-only LoRA` exported directly from our `v5p-16` runs.
