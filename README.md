# `RecursiveSI/flint` (`39.8B` MoE) LoRA & QLoRA Fine-Tuning and Multi-LoRA Serving on Google Cloud TPU

End-to-end **parameter-efficient fine-tuning (`BF16 LoRA`, `INT8 QLoRA`, `NF4 QLoRA`)**, **rank/target ablations (`r=16` vs `r=64`, full vs attention-only)**, **long-context `16K`–`64K` scaling with fused Pallas cross-entropy**, and **Hugging Face `PEFT` Multi-LoRA serving (`SGLang-TorchTPU`)** for **`RecursiveSI/flint`** (`39.81B` total parameters / `2.52B` active parameters per token) on **Google Cloud TPU `v5p-16` (`2x2x4`)** and **TPU `v7x`** using **TorchTitan + TorchTPU `0.2.0` GA**.

---

## Model Architecture & LoRA / QLoRA Quantization Scope

`RecursiveSI/flint` is a `49`-layer sparse Mixture-of-Experts (`MoE`) hybrid-attention decoder (`d_model=2048`, `vocab=150,144`, `39,785,287,456` base parameters):
- **Layers `0..1` (`2` Dense Layers):** Dense SwiGLU MLP (`hidden=6144`) + Full causal Grouped-Query Attention (`NoPE`).
- **Layers `2..48` (`47` Sparse MoE Layers):** **`256` routed top-`6` sigmoid experts** (`expert_hidden=512`, `37.85B` expert parameters) + **`1` ungated shared expert** (`hidden=1024`).
- **Hybrid Attention (`49` Layers):** `35` sliding-window RoPE layers (`W=513`) + `14` full causal `NoPE` layers (`16` query heads, `2` KV heads, `head_dim=128`) with learned per-head softmax sink offsets (`sinks`).

Across our QLoRA recipes ([`flint_lora_tpu_recipes.py`](./finetuning/scripts/flint_lora_tpu_recipes.py)), **both the `390` frozen decoder `nn.Linear` projections (`1.32B` params) AND all `47` `TpuGroupedExperts` `3D` expert weight banks (`w1_EFD, w2_EDF, w3_EFD` = `37.85B` params; `39.17B` quantized parameters total = `98.45%` of the entire model)** are quantized in-place on TPU HBM and sharded across `SimpleFSDP` (`ZeRO-3`) with `param_dtype=None` so **`1`-byte `int8` or packed `4.125`-bit `NF4` shards are all-gathered directly over ICI** without upcasting to `BF16` on the interconnect.

---

## 1. `4K` Fine-Tuning Benchmark Matrix (`TPU v5p-16`, `2x2x4`, `131,072` tok/step)

All five `4K` runs execute on **`16` TPU `v5p` chips** (`4` hosts × `4` chips, `95 GiB` HBM/chip, `459` BF16 TFLOP/s/chip) on packed `yahma/alpaca-cleaned` SFT conversations (`seq_len=4096`, `microbatch=2/chip`, `131,072` global tokens/step, `FullAC` activation checkpointing, `Splash` sink-attention + `Pallas` `256`-expert grouped GEMM `tile_m=256`). Steady-state metrics average steps `15..N`, and runtime HBM is independently verified every `2s` on host `0` via `tpu-info`.

| Recipe & Base Precision | LoRA Config (`Trainable Params`) | Resting Weight Shard / Chip | PyTorch Peak HBM / Chip | `tpu-info` HBM (`p10 / p50 / max`) | Steady Step (`p50`) | Throughput / Chip (`16`-Chip Slice) | Active `MFU` / `HFU` | Loss (`Step 1 -> Final`) | Artifacts |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| **`flint_40b_lora_bf16`** *(Frozen `BF16` Base)* | `r=16, α=32` `wq,wkv,wo,w1,w2,w3` (`21.36M`, `0.054%`) | `4.784 GiB` | `18.326 GiB` | `10.34 / 12.61 / 18.33 GiB` | `2.990 s` | **`2,714.8` tok/s** (`43,436` tok/s) | `11.19%` / `14.92%` | `2.1387 -> 0.8638` *(150 st)* | [`stats`](./finetuning/results/flint_40b_lora_bf16_run_stats.json) · [`hbm`](./finetuning/results/flint_40b_lora_bf16_hbm_telemetry.json) |
| **`flint_40b_qlora_int8`** *(Frozen `INT8` `39.17B` Base)* | `r=16, α=32` `wq,wkv,wo,w1,w2,w3` (`21.36M`, `0.054%`) | **`2.579 GiB` (`-46.1%`)** | **`16.012 GiB` (`-12.6%`)** | **`7.33 / 10.09 / 15.79 GiB`** | `3.140 s` | **`2,586.2` tok/s** (`41,379` tok/s) | `10.66%` / `14.21%` | **`2.1340 -> 0.8633`** *(150 st)* | [`stats`](./finetuning/results/flint_40b_qlora_int8_run_stats.json) · [`hbm`](./finetuning/results/flint_40b_qlora_int8_hbm_telemetry.json) |
| **`flint_40b_qlora_nf4`** *(Frozen `NF4` `4.125b` `39.17B` Base)* | `r=16, α=32` `wq,wkv,wo,w1,w2,w3` (`21.36M`, `0.054%`) | **`1.543 GiB` (`-67.7%`)** | `65.937 GiB` *(expert LUT unpack)* | `7.38 / 29.09 / 65.80 GiB` | `9.155 s` | `892.2` tok/s (`14,275` tok/s) | `3.68%` / `4.90%` | `2.2576 -> 0.8720` *(150 st)* | [`stats`](./finetuning/results/flint_40b_qlora_nf4_run_stats.json) · [`hbm`](./finetuning/results/flint_40b_qlora_nf4_hbm_telemetry.json) |
| **`flint_40b_lora_bf16_r64`** *(High-Rank `r=64` Ablation)* | `r=64, α=128` `wq,wkv,wo,w1,w2,w3` (`85.46M`, `0.214%`) | `4.800 GiB` | `18.383 GiB` | `9.57 / 12.38 / 18.38 GiB` | `3.000 s` | **`2,710.1` tok/s** (`43,362` tok/s) | `11.40%` / `15.20%` | **`2.1387 -> 0.8142`** *(120 st)* | [`stats`](./finetuning/results/flint_40b_lora_bf16_r64_run_stats.json) · [`hbm`](./finetuning/results/flint_40b_lora_bf16_r64_hbm_telemetry.json) |
| **`flint_40b_lora_bf16_attn`** *(Attention-Only Ablation)* | `r=16, α=32` `wq,wkv,wo` (`13.65M`, `0.034%`) | `4.780 GiB` | `18.289 GiB` | `9.49 / 12.71 / 18.29 GiB` | **`2.900 s`** | **`2,811.7` tok/s** (`44,988` tok/s) | **`11.56%` / `15.42%`** | `2.1387 -> 0.9698` *(120 st)* | [`stats`](./finetuning/results/flint_40b_lora_bf16_attn_run_stats.json) · [`hbm`](./finetuning/results/flint_40b_lora_bf16_attn_hbm_telemetry.json) |

### Key Hardware & Systems Findings on Sparse `256`-Expert MoE QLoRA
1. **`INT8` QLoRA Is the Clear TPU Pareto Winner for `256`-Expert MoE Fine-Tuning:** Quantizing `39.17B` base parameters (`37.85B` routed `3D` experts + `1.32B` dense/shared/attention linears, `0.0151` Frobenius relative error) to `int8` cuts resting weight shard HBM by **`46.1%` (`4.784 -> 2.579 GiB/chip`)**, reduces `tpu-info` peak runtime HBM from `18.33 GiB` to **`15.79 GiB` (`-2.54 GiB/chip`)**, retains **`95.3%` of `BF16` LoRA throughput** (`2,586.2` vs `2,714.8` tok/s/chip), and converges to **identical SFT loss** (`0.8633` vs `0.8638` at step `150`). Because TPU `v5p` MXUs execute native `int8 × bf16` / `bf16` Pallas grouped GEMMs after `1`-byte ICI `SimpleFSDP` all-gathers, `INT8` QLoRA saves both resting HBM and transient all-gather buffers simultaneously.
2. **Why `4`-Bit `NF4` Spikes Transient HBM on `256`-Expert MoE (`1.54 GiB` Resting -> `65.80 GiB` Peak):** `NF4` (`4.125` bits/param, block-`64` + `uint8` double-quantized `absmax`) achieves the smallest resting parameter footprint (**`1.543 GiB/chip`, `-67.7%` vs `BF16`**) and converges cleanly (`0.8720` loss). However, unlike `INT8`, TPU MXUs have no native `NF4` codebook matrix-multiply instruction: unpacking `[256, 512, 2048]` `4`-bit indices through a `16`-entry `FP32` lookup tree inside XLA under `FullAC` recomputation materializes transient `~50 GiB` dequantization workspaces (`65.80 GiB` `tpu-info` peak) and runs **`2.90×` slower than `INT8` QLoRA** (`892.2` vs `2,586.2` tok/s/chip).
3. **`4×` Rank Scaling (`r=16 -> r=64`) Is Nearly Free (`-0.17%` tok/s) and Cuts Loss by `-0.050`:** Increasing LoRA rank `4×` (`21.36M -> 85.46M` trainable params) adds only `0.01 s` per step (`2.990 -> 3.000 s`) and `0.05 GiB` HBM while improving `120`-step loss from `0.9293` (`r=16` @ step `120`) to **`0.8142` (`-0.115` at equal step budget)**. Conversely, adapting dense + shared-expert MLPs alongside attention (`flint_40b_lora_bf16` vs `flint_40b_lora_bf16_attn`) improves `120`-step SFT loss from `0.9698` to `0.9293` at a modest `3.4%` throughput cost.
4. **Dual Hugging Face `PEFT` Export (`4.3–4.9 s` Cold Sharded Load, `4.7–12.0 s` Export):** Every run loads `79.6 GB` of `RecursiveSI/flint` safetensors in **`4.3–4.9 s`** by slicing only each rank's local `DTensor` shard (`4.97 GB`/rank, `16` local experts/layer) and exports two ready-to-serve Hugging Face `PEFT` bundles: `peft_adapter` (`343` modules: attention + dense MLP + shared-expert MLP, [`adapter_config_full.json`](./finetuning/results/adapter_config_full.json)) and `peft_adapter_attn` (`196` attention modules across all `49` layers for universal `SGLang-TorchTPU` / `vLLM-TorchTPU` multi-LoRA serving, [`adapter_config_attn.json`](./finetuning/results/adapter_config_attn.json)).

---

## 2. Long-Context LoRA & `INT8` QLoRA Scaling (`16K` -> `64K`, `TPU v5p-16`)

Long-context recipes enable `FusedLossLinearWrapper` + `PallasCrossEntropyLoss` (`FLINT_LORA_PALLAS_CE=1`), deferring the `[seq_len, 150144]` vocabulary projection into a tiled Pallas kernel so the multi-tens-of-GiB `FP32` logits tensor is never materialized in HBM, combined with `Splash` causal/sliding-window sink attention and `Pallas` `256`-expert grouped GEMMs (`tile_m=256`).

| Recipe | Context (`Global Tok/Step`) | Steady Step (`p50`) | Throughput / Chip (`16`-Chip Slice) | Resting Shard HBM | PyTorch Peak HBM | `tpu-info` HBM (`p50 / max`) | Loss (`Step 1 -> Final`) |
|---|---|---:|---:|---:|---:|---:|---:|
| **`flint_40b_lora_bf16_16k`** *(`BF16` LoRA (`r=16`))* | `16K` (`16,384`) (`262,144`) | `5.040 s` | **`3,251.1`** (`52,018.1` tok/s) | `5.236 GiB` | `15.734 GiB` | `11.95 / 15.70 GiB` | `2.1337 -> 0.9506` |
| **`flint_40b_qlora_int8_16k`** *(`INT8` QLoRA (`r=16`))* | `16K` (`16,384`) (`262,144`) | `5.190 s` | **`3,157.5`** (`50,520.8` tok/s) | `3.032 GiB` | `14.015 GiB` | `9.51 / 13.83 GiB` | `2.1294 -> 0.9505` |
| **`flint_40b_lora_bf16_32k`** *(`BF16` LoRA (`r=16`))* | `32K` (`32,768`) (`524,288`) | `7.390 s` | **`4,433.8`** (`70,941.4` tok/s) | `5.807 GiB` | `25.445 GiB` | `18.11 / 25.06 GiB` | `2.1239 -> 0.9553` |
| **`flint_40b_qlora_int8_32k`** *(`INT8` QLoRA (`r=16`))* | `32K` (`32,768`) (`524,288`) | `7.570 s` | **`4,330.0`** (`69,280.0` tok/s) | `3.603 GiB` | `21.729 GiB` | `14.34 / 21.04 GiB` | `2.1210 -> 0.9548` |
| **`flint_40b_lora_bf16_64k`** *(`BF16` LoRA (`r=16`))* | `64K` (`65,536`) (`1,048,576`) | `12.130 s` | **`5,402.5`** (`86,440.4` tok/s) | `6.931 GiB` | `44.175 GiB` | `29.61 / 43.40 GiB` | `2.1172 -> 0.9539` |
| **`flint_40b_qlora_int8_64k`** *(`INT8` QLoRA (`r=16`))* | `64K` (`65,536`) (`1,048,576`) | `12.370 s` | **`5,297.8`** (`84,765.0` tok/s) | `4.727 GiB` | `41.820 GiB` | `27.26 / 41.06 GiB` | `2.1144 -> 0.9535` |


---

## Repository Structure

- [`docs/FLINT_40B_TPU_LORA_QLORA_REPORT.md`](./docs/FLINT_40B_TPU_LORA_QLORA_REPORT.md) — Deep-dive technical report: architecture mapping, `INT8` vs `NF4` Pallas grouped-expert quantization mechanics, `FusedLossLinearWrapper` + `PallasCrossEntropyLoss` long-context HBM analysis, `tpu-info` telemetry, and `SGLang-TorchTPU` multi-LoRA runtime patches.
- [`finetuning/scripts/`](./finetuning/scripts/)
  - [`flint_lora_tpu_recipes.py`](./finetuning/scripts/flint_lora_tpu_recipes.py) — All `4K` (`BF16`, `INT8`, `NF4`, `r=64`, `attn-only`) and `16K / 32K / 64K` long-context recipes, sharded safetensors loader, `INT8`/`NF4` `TpuGroupedExperts` quantizers, and dual `PEFT` exporter.
  - [`launch_flint_ft.sh`](./finetuning/scripts/launch_flint_ft.sh), [`ft_flint_ctl.py`](./finetuning/scripts/ft_flint_ctl.py), [`hbm_sampler.sh`](./finetuning/scripts/hbm_sampler.sh), [`parse_flint_ft_results.py`](./finetuning/scripts/parse_flint_ft_results.py) — Distributed `4`-host launcher, multi-pod orchestrator, `2s` `tpu-info` HBM daemon, and telemetry parser.
- [`finetuning/k8s/flint-lora-ft-v5p16.yaml`](./finetuning/k8s/flint-lora-ft-v5p16.yaml) — Indexed `4`-host `TPU v5p-16` (`2x2x4`) Kubernetes Job manifest.
- [`finetuning/results/`](./finetuning/results/) — [`summary.json`](./finetuning/results/summary.json), per-run `*_run_stats.json`, `*_hbm_telemetry.json`, [`adapter_config_full.json`](./finetuning/results/adapter_config_full.json), and [`adapter_config_attn.json`](./finetuning/results/adapter_config_attn.json).
- [`serving/scripts/`](./serving/scripts/) — [`patch_sglang_tpu_lora.py`](./serving/scripts/patch_sglang_tpu_lora.py) (automated `5`-gap `SGLang-TorchTPU` `FlintForCausalLM` LoRA enabler for `BF16` & `Block-FP8` bases), [`bench_flint_multilora_sglang.py`](./serving/scripts/bench_flint_multilora_sglang.py), and [`bench_flint_vllm_rqkr.py`](./serving/scripts/bench_flint_vllm_rqkr.py).
- [`serving/k8s/`](./serving/k8s/) — [`flint-multilora-sglang-v7x.yaml`](./serving/k8s/flint-multilora-sglang-v7x.yaml) and [`flint-serve-vllm-v7x.yaml`](./serving/k8s/flint-serve-vllm-v7x.yaml).

---

## Quick Reproduction

### 1. Multi-Host `TPU v5p-16` LoRA & QLoRA Fine-Tuning
```bash
kubectl apply -f finetuning/k8s/flint-lora-ft-v5p16.yaml
python3 finetuning/scripts/ft_flint_ctl.py suite40b
python3 finetuning/scripts/ft_flint_ctl.py suite_longctx
```

### 2. `SGLang-TorchTPU` Multi-LoRA Serving (`BF16` & `Block-FP8` Base + Exported `PEFT` Adapters)
```bash
kubectl apply -f serving/k8s/flint-multilora-sglang-v7x.yaml
python3 serving/scripts/bench_flint_multilora_sglang.py
```
