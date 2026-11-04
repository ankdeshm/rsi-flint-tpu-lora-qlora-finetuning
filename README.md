# `RecursiveSI/flint` (`39.8B` MoE) LoRA & QLoRA Fine-Tuning (`4K`–`128K`) and Multi-LoRA Serving on Google Cloud TPU

End-to-end **parameter-efficient fine-tuning (`BF16 LoRA`, `INT8 QLoRA`, `NF4 QLoRA`)**, **rank/target ablations (`r=16` vs `r=64`, full vs attention-only)**, **long-context & ultra-long-context `16K`–`128K` (`seq_len = 131,072`) scaling with fused Pallas cross-entropy**, and **Hugging Face `PEFT` Multi-LoRA serving (`SGLang-TorchTPU`)** for **`RecursiveSI/flint`** (`39.81B` total parameters / `2.52B` active parameters per token) on **Google Cloud TPU `v5p-16` (`2x2x4`)** and **TPU `v7x`** using **TorchTitan + TorchTPU `0.2.0` GA**.

---

## Model Architecture & LoRA / QLoRA Quantization Scope

`RecursiveSI/flint` is a `49`-layer sparse Mixture-of-Experts (`MoE`) hybrid-attention decoder (`d_model=2048`, `vocab=150,144`, `39,785,287,456` base parameters):
- **Layers `0..1` (`2` Dense Layers):** Dense SwiGLU MLP (`hidden=6144`) + Full causal Grouped-Query Attention (`NoPE`).
- **Layers `2..48` (`47` Sparse MoE Layers):** **`256` routed top-`6` sigmoid experts** (`expert_hidden=512`, `37.85B` expert parameters) + **`1` ungated shared expert** (`hidden=1024`).
- **Hybrid Attention (`49` Layers):** `35` sliding-window RoPE layers (`W=513`) + `14` full causal `NoPE` layers (`32` query heads, `4` KV heads, `8:1` GQA, `head_dim=128`) with learned per-head softmax sink offsets (`sinks`).

Across our QLoRA recipes ([`flint_lora_tpu_recipes.py`](./finetuning/scripts/flint_lora_tpu_recipes.py)), **both the `390` frozen decoder `nn.Linear` projections (`1.32B` params) AND all `47` `TpuGroupedExperts` `3D` expert weight banks (`w1_EFD, w2_EDF, w3_EFD` = `37.85B` params; `39.17B` quantized parameters total = `98.45%` of the entire model)** are quantized in-place on TPU HBM and sharded across `SimpleFSDP` (`ZeRO-3`) with `param_dtype=None` so **`1`-byte `int8` or packed `4.125`-bit `NF4` shards are all-gathered directly over ICI** without upcasting to `BF16` on the interconnect.

---

## 1. `4K` Fine-Tuning Benchmark Matrix (`TPU v5p-16`, `2x2x4`, `131,072` tok/step)

All five `4K` runs execute on **`16` TPU `v5p` chips** (`4` hosts × `4` chips, `95 GiB` HBM/chip, `459` BF16 TFLOP/s/chip) on packed `yahma/alpaca-cleaned` SFT conversations (`seq_len=4096`, `microbatch=2/chip`, `131,072` global tokens/step, `FullAC` activation checkpointing, `Splash` sink-attention + `Pallas` `256`-expert grouped GEMM `tile_m=256`). Steady-state metrics average steps `15..N`, and runtime HBM is independently verified every `2s` on host `0` via `tpu-info`.

| Recipe & Base Precision | LoRA Config (`Trainable Params`) | Resting Weight Shard / Chip | `torch_tpu` Peak HBM / Chip | `tpu-info` HBM (`p10 / p50 / max`) | Steady Step (`p50`) | Throughput / Chip (`16`-Chip Slice) | TorchTitan `MFU` / `HFU` *(6N ref)* | Loss (`Step 1 -> Final`) | Artifacts |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| **`flint_40b_lora_bf16`** *(Frozen `BF16` Base)* | `r=16, α=32` `wq,wkv,wo,w1,w2,w3` (`21.36M`, `0.054%`) | `4.784 GiB` | `18.326 GiB` | `10.34 / 12.61 / 18.33 GiB` | `2.990 s` | **`2,714.8` tok/s** (`43,436` tok/s) | `11.19%` / `14.92%` | `2.1387 -> 0.8638` *(150 st)* | [`stats`](./finetuning/results/flint_40b_lora_bf16_run_stats.json) · [`hbm`](./finetuning/results/flint_40b_lora_bf16_hbm_telemetry.json) |
| **`flint_40b_qlora_int8`** *(Frozen `INT8` `39.17B` Base)* | `r=16, α=32` `wq,wkv,wo,w1,w2,w3` (`21.36M`, `0.054%`) | **`2.579 GiB` (`-46.1%`)** | **`16.012 GiB` (`-12.6%`)** | **`7.33 / 10.09 / 15.79 GiB`** | `3.140 s` | **`2,586.2` tok/s** (`41,379` tok/s) | `10.66%` / `14.21%` | **`2.1340 -> 0.8633`** *(150 st)* | [`stats`](./finetuning/results/flint_40b_qlora_int8_run_stats.json) · [`hbm`](./finetuning/results/flint_40b_qlora_int8_hbm_telemetry.json) |
| **`flint_40b_qlora_nf4`** *(Frozen `NF4` `4.125b` `39.17B` Base)* | `r=16, α=32` `wq,wkv,wo,w1,w2,w3` (`21.36M`, `0.054%`) | **`1.543 GiB` (`-67.7%`)** | `65.937 GiB` *(expert LUT unpack)* | `7.38 / 29.09 / 65.80 GiB` | `9.155 s` | `892.2` tok/s (`14,275` tok/s) | `3.68%` / `4.90%` | `2.2576 -> 0.8720` *(150 st)* | [`stats`](./finetuning/results/flint_40b_qlora_nf4_run_stats.json) · [`hbm`](./finetuning/results/flint_40b_qlora_nf4_hbm_telemetry.json) |
| **`flint_40b_lora_bf16_r64`** *(High-Rank `r=64` Ablation)* | `r=64, α=128` `wq,wkv,wo,w1,w2,w3` (`85.46M`, `0.214%`) | `4.800 GiB` | `18.383 GiB` | `9.57 / 12.38 / 18.38 GiB` | `3.000 s` | **`2,710.1` tok/s** (`43,362` tok/s) | `11.40%` / `15.20%` | **`2.1387 -> 0.8142`** *(120 st)* | [`stats`](./finetuning/results/flint_40b_lora_bf16_r64_run_stats.json) · [`hbm`](./finetuning/results/flint_40b_lora_bf16_r64_hbm_telemetry.json) |
| **`flint_40b_lora_bf16_attn`** *(Attention-Only Ablation)* | `r=16, α=32` `wq,wkv,wo` (`13.65M`, `0.034%`) | `4.780 GiB` | `18.289 GiB` | `9.49 / 12.71 / 18.29 GiB` | **`2.900 s`** | **`2,811.7` tok/s** (`44,988` tok/s) | **`11.56%` / `15.42%`** | `2.1387 -> 0.9698` *(120 st)* | [`stats`](./finetuning/results/flint_40b_lora_bf16_attn_run_stats.json) · [`hbm`](./finetuning/results/flint_40b_lora_bf16_attn_hbm_telemetry.json) |

### Key Hardware & Systems Findings on Sparse `256`-Expert MoE QLoRA
1. **`INT8` QLoRA Is the Clear TPU Pareto Winner for `256`-Expert MoE Fine-Tuning:** Quantizing `39.17B` base parameters (`37.85B` routed `3D` experts + `1.32B` dense/shared/attention linears, `0.0151` Frobenius relative error) to `int8` cuts resting weight shard HBM by **`46.1%` (`4.784 -> 2.579 GiB/chip`)**, reduces `tpu-info` peak runtime HBM from `18.33 GiB` to **`15.79 GiB` (`-2.54 GiB/chip`)**, retains **`95.3%` of `BF16` LoRA throughput** (`2,586.2` vs `2,714.8` tok/s/chip), and converges to **identical SFT loss** (`0.8633` vs `0.8638` at step `150`). Because TPU `v5p` MXUs execute native `int8 × bf16` / `bf16` Pallas grouped GEMMs after `1`-byte ICI `SimpleFSDP` all-gathers, `INT8` QLoRA saves both resting HBM and transient all-gather buffers simultaneously.
2. **Why `4`-Bit `NF4` Spikes Transient HBM on `256`-Expert MoE (`1.54 GiB` Resting -> `65.80 GiB` Peak):** `NF4` (`4.125` bits/param, block-`64` + `uint8` double-quantized `absmax`) achieves the smallest resting parameter footprint (**`1.543 GiB/chip`, `-67.7%` vs `BF16`**) and converges cleanly (`0.8720` loss). However, unlike `INT8`, TPU MXUs have no native `NF4` codebook matrix-multiply instruction: unpacking `[256, 512, 2048]` `4`-bit indices inside XLA under `FullAC` recomputation materializes transient dequantization workspaces (`65.80 GiB` `tpu-info` peak on the `4K` `FP32` select-tree run; [`flint_lora_tpu_recipes.py`](./finetuning/scripts/flint_lora_tpu_recipes.py) further replaces the `FP32` binary tree with a cached `BF16` `16`-entry LUT gather `lut[idx.to(torch.int32)]` for `128K` runs) and runs **`2.90×` slower than `INT8` QLoRA** (`892.2` vs `2,586.2` tok/s/chip).
3. **`4×` Rank Scaling (`r=16 -> r=64`) Is Nearly Free (`-0.17%` tok/s) and Cuts Loss by `-0.050`:** Increasing LoRA rank `4×` (`21.36M -> 85.46M` trainable params) adds only `0.01 s` per step (`2.990 -> 3.000 s`) and `0.05 GiB` HBM while improving `120`-step loss from `0.9293` (`r=16` @ step `120`) to **`0.8142` (`-0.115` at equal step budget)**. Conversely, adapting dense + shared-expert MLPs alongside attention (`flint_40b_lora_bf16` vs `flint_40b_lora_bf16_attn`) improves `120`-step SFT loss from `0.9698` to `0.9293` at a modest `3.4%` throughput cost.
4. **Dual Hugging Face `PEFT` Export (`4.3–5.7 s` `/dev/shm` Sharded Load, `4.7–12.0 s` Export):** Every run slices only each rank's local `DTensor` shard (`4.97 GB`/rank, `16` local experts/layer) and exports two ready-to-serve Hugging Face `PEFT` bundles: `peft_adapter` (`343` modules: attention + dense MLP + shared-expert MLP, [`adapter_config_full.json`](./finetuning/results/adapter_config_full.json)) and `peft_adapter_attn` (`196` attention modules across all `49` layers for universal `SGLang-TorchTPU` / `vLLM-TorchTPU` multi-LoRA serving, [`adapter_config_attn.json`](./finetuning/results/adapter_config_attn.json)).
5. **Measurement Conventions (`HBM` & `MFU`):** `torch_tpu` `peak_allocated_gib` (`mem_end.peak_allocated_gib` in `*_run_stats.json`) tracks `tpu-info` `max_gib` within `<2%` across all `tpu-info`-instrumented runs (`18.326` vs `18.33 GiB` on `4K BF16`, `16.012` vs `15.79 GiB` on `4K INT8`, `44.175` vs `43.40 GiB` on `64K BF16`), whereas TorchTitan's step-line `memory:` field logs `reserved_bytes.all.peak` rather than active device HBM. TorchTitan's logged `mfu` uses the standard `6 * N_active + 12 * L * d_model * S` full-fine-tuning reference formula (`HFU = 4/3 * MFU` under `FullAC`); use `tok/s/chip` and step latency for direct cross-precision and cross-context (`4K`–`128K`) comparisons.

---

## 2. Long-Context & Ultra-Long-Context LoRA / `INT8` QLoRA Scaling (`16K` -> `128K`, `TPU v5p-16`)

Long-context recipes (`seq_len >= 16,384`, up to **`131,072` (`128K`, `2,097,152` global tokens/step)**) enable `FusedLossLinearWrapper` + `PallasCrossEntropyLoss` (`FLINT_LORA_PALLAS_CE=1`), deferring the `[seq_len, 150144]` vocabulary projection into a tiled Pallas VMEM kernel (`b_block=1024, h_block=512, v_block=2048`) so the `78.7–157.4 GB` `FP32` logits + gradient tensors are never materialized in HBM, combined with `Splash` causal/sliding-window sink attention and `Pallas` `256`-expert grouped GEMMs (`tile_m=256`).

| Recipe | Context (`Global Tok/Step`) | Steady Step (`p50`) | Throughput / Chip (`16`-Chip Slice) | Resting Shard HBM | `torch_tpu` Peak HBM | `tpu-info` HBM (`p50 / max`) | Loss (`Step 1 -> Final`) |
|---|---|---:|---:|---:|---:|---:|---:|
| **`flint_40b_lora_bf16_16k`** *(`BF16` LoRA (`r=16`))* | `16K` (`16,384`) (`262,144`) | `5.040 s` | **`3,251.1`** (`52,018.1` tok/s) | `5.236 GiB` | `15.734 GiB` | `11.95 / 15.70 GiB` | `2.1337 -> 0.9506` |
| **`flint_40b_qlora_int8_16k`** *(`INT8` QLoRA (`r=16`))* | `16K` (`16,384`) (`262,144`) | `5.190 s` | **`3,157.5`** (`50,520.8` tok/s) | `3.032 GiB` | `14.015 GiB` | `9.51 / 13.83 GiB` | `2.1294 -> 0.9505` |
| **`flint_40b_lora_bf16_32k`** *(`BF16` LoRA (`r=16`))* | `32K` (`32,768`) (`524,288`) | `7.390 s` | **`4,433.8`** (`70,941.4` tok/s) | `5.807 GiB` | `25.445 GiB` | `18.11 / 25.06 GiB` | `2.1239 -> 0.9553` |
| **`flint_40b_qlora_int8_32k`** *(`INT8` QLoRA (`r=16`))* | `32K` (`32,768`) (`524,288`) | `7.570 s` | **`4,330.0`** (`69,280.0` tok/s) | `3.603 GiB` | `21.729 GiB` | `14.34 / 21.04 GiB` | `2.1210 -> 0.9548` |
| **`flint_40b_lora_bf16_64k`** *(`BF16` LoRA (`r=16`))* | `64K` (`65,536`) (`1,048,576`) | `12.130 s` | **`5,402.5`** (`86,440.4` tok/s) | `6.931 GiB` | `44.175 GiB` | `29.61 / 43.40 GiB` | `2.1172 -> 0.9539` |
| **`flint_40b_qlora_int8_64k`** *(`INT8` QLoRA (`r=16`))* | `64K` (`65,536`) (`1,048,576`) | `12.370 s` | **`5,297.8`** (`84,765.0` tok/s) | `4.727 GiB` | `41.820 GiB` | `27.26 / 41.06 GiB` | `2.1144 -> 0.9535` |
| **`flint_40b_lora_bf16_128k`** *(`BF16` LoRA (`r=16`))* | `128K` (`131,072`) (`2,097,152`) | `45.235 s` | **`2,897.6`** (`46,361.0` tok/s) | `9.244 GiB` | `81.242 GiB` | *(`torch_tpu`: `81.24 GiB`)* | `2.1087 -> 0.9900` |

- **Why `16K -> 64K` Doubles Throughput (`2,714.8 -> 5,402.5 tok/s/chip`) and `128K` Still Beats `4K` (`2,897.6 tok/s/chip`):** From `4K` to `64K`, larger token batches per `256`-expert Pallas grouped GEMM (`tile_m=256`) and fewer `SimpleFSDP` ICI `AllGather` collectives per million tokens double per-chip throughput. At **`128K` (`seq_len = 131,072`)**, `RecursiveSI/flint`'s `14` full-sequence `NoPE` causal layers enter the quadratic `O(S^2)` Splash-attention regime while all `35` sliding-window (`W=513`) layers stay `O(S · W)` linear—delivering **`2,897.6 tok/s/chip` (`46,361 tok/s` across `16` `v5p` chips, `+6.7%` higher than `4K`)** and **`81.242 GiB` peak HBM (`85.5%` of `95 GiB` `v5p` HBM, zero context parallelism or host offload required)** while converging cleanly from `2.1087 -> 0.9901` in `20` steps (`41.94M` tokens).

---

## 3. Compiled Multi-LoRA & QLoRA-Style Serving (`SGLang-TorchTPU` on TPU `v7x` & `v5p`)

### 3.1 TPU `v7x` Single-Host (`4` Chips / `8` TensorCores, `TP=2 × DP=4`) Compiled Multi-LoRA Serving

Artifacts: [`serving/results/v7x_multilora_summary.json`](./serving/results/v7x_multilora_summary.json) · [`serving/scripts/patch_sglang_tpu_lora.py`](./serving/scripts/patch_sglang_tpu_lora.py) · [`serving/scripts/bench_flint_multilora_sglang.py`](./serving/scripts/bench_flint_multilora_sglang.py)

#### `E1_sglang_fp8_multilora` (`fp8_block128` Base + `FP8` KV) — Dynamic `/unload_lora_adapter` + `/load_lora_adapter`: **`13.3 ms` unload / `24.8 ms` reload** (`0` errors)

| Adapter Schedule | `1K` `c=1` (`TPOT p50`) | `1K` Peak Host (`4`-Host Equiv) | `4K` Peak Host | `64K` Peak Host | Failed |
|---|---:|---:|---:|---:|---:|
| `0` LoRAs *(Base Only)* | `95.5` tok/s (`10.47 ms`) | **`3,984.6`** (`15,938.5` tok/s) | `3,487.7` tok/s | `434.4` tok/s | `0` |
| `1` LoRA *(`lora_bf16`)* | `88.2` tok/s (`11.34 ms`) | **`4,133.8`** (`16,535.4` tok/s) | `3,590.7` tok/s | `436.9` tok/s | `0` |
| `2` Concurrent *(`BF16 + INT8 QLoRA`)* | `90.9` tok/s (`11.00 ms`) | **`4,322.8`** (`17,291.4` tok/s) | — | — | `0` |
| `4` Concurrent *(`BF16 + INT8 + NF4 + Attn`)* | `91.3` tok/s (`10.95 ms`) | **`3,634.9`** (`14,539.8` tok/s) | `3,001.4` tok/s | `365.2` tok/s | `0` |
| Mixed *(`Base + 3 LoRAs`)* | `92.3` tok/s (`10.84 ms`) | **`4,352.3`** (`17,409.3` tok/s) | — | — | `0` |

#### `E2_sglang_bf16_multilora` (`bfloat16` Base + `FP8` KV) — Dynamic `/unload_lora_adapter` + `/load_lora_adapter`: **`12.4 ms` unload / `24.4 ms` reload** (`0` errors)

| Adapter Schedule | `1K` `c=1` (`TPOT p50`) | `1K` Peak Host (`4`-Host Equiv) | `4K` Peak Host | `64K` Peak Host | Failed |
|---|---:|---:|---:|---:|---:|
| `0` LoRAs *(Base Only)* | `87.7` tok/s (`11.40 ms`) | **`4,120.3`** (`16,481.2` tok/s) | `3,603.9` tok/s | `431.3` tok/s | `0` |
| `1` LoRA *(`lora_bf16`)* | `87.5` tok/s (`11.43 ms`) | **`4,293.8`** (`17,175.2` tok/s) | `3,573.5` tok/s | `440.0` tok/s | `0` |
| `2` Concurrent *(`BF16 + INT8 QLoRA`)* | `87.7` tok/s (`11.40 ms`) | **`4,175.2`** (`16,700.7` tok/s) | — | — | `0` |
| `4` Concurrent *(`BF16 + INT8 + NF4 + Attn`)* | `90.6` tok/s (`11.03 ms`) | **`3,633.6`** (`14,534.2` tok/s) | `3,055.4` tok/s | `365.4` tok/s | `0` |
| Mixed *(`Base + 3 LoRAs`)* | `91.2` tok/s (`10.96 ms`) | **`4,162.7`** (`16,650.9` tok/s) | — | — | `0` |

### 3.2 TPU `v5p` Single-Host (`4` Chips, `TP=4 × DP=1`, `--lora-backend torch_native`) Compiled Multi-LoRA Serving (`16`-Chip Slice Equiv)

Artifacts: [`serving/results/v5p_multilora_summary.json`](./serving/results/v5p_multilora_summary.json) · [`serving/scripts/run_v5p_wave_b_worker.py`](./serving/scripts/run_v5p_wave_b_worker.py) · [`serving/k8s/flint-multilora-sglang-v5p.yaml`](./serving/k8s/flint-multilora-sglang-v5p.yaml)

#### `v5p_fp8_block128_multilora` (`fp8_block128` Base + `FP8` KV) — Dynamic hot-swap: **`22 ms` unload / `26 ms` reload**

| Adapter Schedule | `1K` `c=1` (`TPOT p50`) | `1K` Peak Host (`4`-Host Equiv) | `4K` Peak Host | `64K` Peak Host | Failed |
|---|---:|---:|---:|---:|---:|
| `0` LoRAs *(Base Only)* | `66.2` tok/s (`15.12 ms`) | **`604.5`** (`2,418.1` tok/s) | `728.7` tok/s | `244.4` tok/s | `0` |
| `1` LoRA *(`lora_bf16`)* | `69.1` tok/s (`14.48 ms`) | **`748.0`** (`2,991.8` tok/s) | `952.0` tok/s | `263.4` tok/s | `0` |
| `2` Concurrent *(`BF16 + INT8 QLoRA`)* | `69.0` tok/s (`14.49 ms`) | **`803.7`** (`3,214.8` tok/s) | — | — | `0` |
| `4` Concurrent *(`BF16 + INT8 + NF4 + Attn`)* | `67.3` tok/s (`14.87 ms`) | **`741.5`** (`2,966.2` tok/s) | `860.5` tok/s | `130.6` tok/s | `0` |
| Mixed *(`Base + 3 LoRAs`)* | `70.3` tok/s (`14.22 ms`) | **`679.7`** (`2,718.6` tok/s) | — | — | `0` |

#### `v5p_bfloat16_multilora` (`bfloat16` Base + `FP8` KV) — Dynamic hot-swap: **`20 ms` unload / `28 ms` reload**

| Adapter Schedule | `1K` `c=1` (`TPOT p50`) | `1K` Peak Host (`4`-Host Equiv) | `4K` Peak Host | `64K` Peak Host | Failed |
|---|---:|---:|---:|---:|---:|
| `0` LoRAs *(Base Only)* | `69.6` tok/s (`14.37 ms`) | **`1,529.1`** (`6,116.4` tok/s) | `1,620.0` tok/s | `385.3` tok/s | `0` |
| `1` LoRA *(`lora_bf16`)* | `67.0` tok/s (`14.92 ms`) | **`1,540.9`** (`6,163.6` tok/s) | `1,647.6` tok/s | `393.1` tok/s | `0` |
| `2` Concurrent *(`BF16 + INT8 QLoRA`)* | `68.3` tok/s (`14.63 ms`) | **`1,655.9`** (`6,623.6` tok/s) | — | — | `0` |
| `4` Concurrent *(`BF16 + INT8 + NF4 + Attn`)* | `68.7` tok/s (`14.55 ms`) | **`1,632.1`** (`6,528.5` tok/s) | `1,580.2` tok/s | `231.8` tok/s | `0` |
| Mixed *(`Base + 3 LoRAs`)* | `68.7` tok/s (`14.55 ms`) | **`1,577.8`** (`6,311.4` tok/s) | — | — | `0` |

### 3.3 Empirical Engine Comparison: `SGLang-TorchTPU` vs `vLLM-TorchTPU` on `RecursiveSI/flint`

- **Stock Upstream `vllm-torchtpu` Probe ([`serving/results/vllm_flint_comparison.json`](./serving/results/vllm_flint_comparison.json)):** `--model-impl auto` -> `exited`, `--model-impl transformers` -> `exited` (`FlintForCausalLM` / `model_type='flint'` is unregistered in `vLLM`'s `ModelRegistry`, `flint_for_causal_lm_registered=False`, and the checkpoint ships `flint_sglang` without a Hugging Face `auto_map`).
- **Plugin-Enabled `vllm-torchtpu` Multi-LoRA Constraint:** Even after registering `FlintConfig` + `FlintForCausalLM` via [`install_flint_vllm.py`](./serving/scripts/install_flint_vllm.py), `VllmConfig` explicitly rejects `--enable-lora` on TPU for both `TP=2 × DP=4 × EP=8` and `TP=1 × DP=8 × EP=8` with `ValidationError: LoRA on TPU does not support tensor_parallel_size > 1; data_parallel_size > 1; MoE models`. By contrast, [`patch_sglang_tpu_lora.py`](./serving/scripts/patch_sglang_tpu_lora.py) enables static-shape XLA-compiled `TorchNativeLoRABackend` Multi-LoRA across `TP=2 × DP=4` (`v7x`) and `TP=4 × DP=1` (`v5p`) on `RecursiveSI/flint`'s `256`-expert MoE with `0` errors on both `BF16` and `Block-FP8` `[128, 128]` base weights.

Full single-host & multi-slice serving optimization matrices (`Axes A–E`, `128K` RadixCache, `swa_ratio`, `TP × DP × EP`) are published in the companion serving repository: [`ankdeshm/rsi-flint-torchtpu-tpu-serving`](https://github.com/ankdeshm/rsi-flint-torchtpu-tpu-serving) (`docs/SINGLEHOST_AND_MULTILORA_OPTIMIZATION.md` and `multilora_singlehost_opt/`).

---

## Repository Structure

- [`docs/FLINT_40B_TPU_LORA_QLORA_REPORT.md`](./docs/FLINT_40B_TPU_LORA_QLORA_REPORT.md) — Deep-dive technical report: architecture mapping, `INT8` vs `NF4` Pallas grouped-expert quantization mechanics, `FusedLossLinearWrapper` + `PallasCrossEntropyLoss` `16K`–`128K` HBM analysis, `tpu-info` telemetry, and `SGLang-TorchTPU` multi-LoRA runtime patches.
- [`finetuning/scripts/`](./finetuning/scripts/)
  - [`flint_lora_tpu_recipes.py`](./finetuning/scripts/flint_lora_tpu_recipes.py) — All `4K` (`BF16`, `INT8`, `NF4`, `r=64`, `attn-only`) and `16K / 32K / 64K / 128K` long-context recipes, sharded safetensors loader, `INT8`/`NF4` `TpuGroupedExperts` quantizers, and dual `PEFT` exporter.
  - [`launch_flint_ft.sh`](./finetuning/scripts/launch_flint_ft.sh), [`ft_flint_ctl.py`](./finetuning/scripts/ft_flint_ctl.py), [`hbm_sampler.sh`](./finetuning/scripts/hbm_sampler.sh), [`parse_flint_ft_results.py`](./finetuning/scripts/parse_flint_ft_results.py) — Distributed `4`-host launcher, multi-pod orchestrator, `2s` `tpu-info` HBM daemon, and telemetry parser.
- [`finetuning/k8s/flint-lora-ft-v5p16.yaml`](./finetuning/k8s/flint-lora-ft-v5p16.yaml) — Indexed `4`-host `TPU v5p-16` (`2x2x4`) Kubernetes Job manifest.
- [`finetuning/results/`](./finetuning/results/) — [`summary.json`](./finetuning/results/summary.json) (`12` complete `4K`–`128K` runs), per-run `*_run_stats.json`, `*_hbm_telemetry.json`, [`adapter_config_full.json`](./finetuning/results/adapter_config_full.json), and [`adapter_config_attn.json`](./finetuning/results/adapter_config_attn.json).
- [`serving/scripts/`](./serving/scripts/) — [`patch_sglang_tpu_lora.py`](./serving/scripts/patch_sglang_tpu_lora.py) (automated `5`-gap `SGLang-TorchTPU` `FlintForCausalLM` LoRA enabler for `BF16` & `Block-FP8` bases up to `128K` context), [`bench_flint_multilora_sglang.py`](./serving/scripts/bench_flint_multilora_sglang.py), and [`bench_flint_vllm_rqkr.py`](./serving/scripts/bench_flint_vllm_rqkr.py).
- [`serving/k8s/`](./serving/k8s/) — [`flint-multilora-sglang-v7x.yaml`](./serving/k8s/flint-multilora-sglang-v7x.yaml) and [`flint-serve-vllm-v7x.yaml`](./serving/k8s/flint-serve-vllm-v7x.yaml).

---

## Quick Reproduction

### 1. Multi-Host `TPU v5p-16` LoRA & QLoRA Fine-Tuning (`4K`, `16K–64K`, and `128K`)
```bash
kubectl apply -f finetuning/k8s/flint-lora-ft-v5p16.yaml
python3 finetuning/scripts/ft_flint_ctl.py suite40b
python3 finetuning/scripts/ft_flint_ctl.py suite_longctx
python3 finetuning/scripts/ft_flint_ctl.py suite_128k
```

### 2. `SGLang-TorchTPU` Multi-LoRA Serving (`BF16` & `Block-FP8` Base + Exported `PEFT` Adapters)
```bash
kubectl apply -f serving/k8s/flint-multilora-sglang-v7x.yaml
python3 serving/scripts/bench_flint_multilora_sglang.py
```
