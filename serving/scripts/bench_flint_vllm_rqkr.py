#!/usr/bin/env python3
"""Empirical vllm-torchtpu vs sglang-torchtpu Engine Comparison & Compatibility Probe for RecursiveSI/flint on TPU v7x."""

from __future__ import annotations

import asyncio
import importlib.metadata
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any


MODEL_DIR = "/models/models/RecursiveSI/flint"
OUT_JSON = Path("/models/benchmark-results/flint_v7x_rqkr_opt/runs/E3_vllm_torchtpu_flint_comparison.json")


def dist_ver(pkg: str) -> str | None:
    try:
        return importlib.metadata.version(pkg)
    except Exception:
        return None


def http_ok(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2.0) as r:
            return r.status == 200
    except Exception:
        return False


def probe_vllm_launch(tag: str, extra_flags: list[str], port: int = 8100, timeout_s: int = 240) -> dict[str, Any]:
    log_path = Path(f"/tmp/vllm_{tag}.log")
    if log_path.exists():
        log_path.unlink()
    env = os.environ.copy()
    env["PJRT_DEVICE"] = "TPU"
    cmd = [
        "vllm",
        "serve",
        MODEL_DIR,
        "--port",
        str(port),
        "--tensor-parallel-size",
        "2",
        "--max-model-len",
        "8192",
        "--gpu-memory-utilization",
        "0.80",
        "--trust-remote-code",
        *extra_flags,
    ]
    t0 = time.time()
    with log_path.open("wb") as lf:
        proc = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
    status = "timeout"
    rc = None
    while time.time() - t0 < timeout_s:
        if http_ok(port):
            status = "ready"
            break
        rc = proc.poll()
        txt = log_path.read_text(errors="replace") if log_path.exists() else ""
        if rc is not None:
            status = "exited"
            break
        if "Traceback (most recent call last):" in txt and any(
            err in txt for err in ("ValueError:", "KeyError:", "NotImplementedError:", "RuntimeError:", "ImportError:", "AttributeError:")
        ):
            time.sleep(2.0)
            rc = proc.poll()
            status = "failed_exception"
            break
        time.sleep(2.0)

    elapsed = round(time.time() - t0, 2)
    try:
        os.killpg(proc.pid, 9)
    except Exception:
        pass

    full_log = log_path.read_text(errors="replace") if log_path.exists() else ""
    lines = full_log.splitlines()
    # Extract key error lines
    err_lines = [
        ln for ln in lines
        if any(k in ln for k in ("Error", "Exception", "Unsupported", "FlintForCausalLM", "flint", "NotImplemented"))
    ][-25:]
    return {
        "tag": tag,
        "command": " ".join(cmd),
        "status": status,
        "returncode": rc,
        "elapsed_sec": elapsed,
        "error_highlights": err_lines,
        "log_tail": "\n".join(lines[-45:]),
    }


def main() -> None:
    # Check if RecursiveSI/flint ships any vllm wheel vs sglang wheel
    model_p = Path(MODEL_DIR)
    shipped_artifacts = sorted(str(p.relative_to(model_p)) for p in model_p.rglob("*.whl")) if model_p.exists() else []
    cfg = json.loads((model_p / "config.json").read_text()) if (model_p / "config.json").exists() else {}

    supported_archs: list[str] = []
    has_flint_arch = False
    try:
        from vllm.model_executor.models import ModelRegistry

        supported_archs = sorted(ModelRegistry.get_supported_archs())
        has_flint_arch = "FlintForCausalLM" in supported_archs
    except Exception as e:
        supported_archs = [f"error: {e}"]

    # Install flint_sglang or check auto_map in config.json
    p_auto = probe_vllm_launch("native_auto", ["--model-impl", "auto"], port=8101, timeout_s=180)
    p_tf = probe_vllm_launch("transformers_backend", ["--model-impl", "transformers"], port=8102, timeout_s=180)

    report = {
        "run_name": "E3_vllm_torchtpu_flint_comparison",
        "axis": "E_engine_comparison",
        "host": os.uname().nodename,
        "hardware": f"Single-Host TPU v7x ({os.uname().nodename}, 4 chips = 8 TensorCores)",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "versions": {
            "vllm": dist_ver("vllm"),
            "torch": dist_ver("torch"),
            "torch_tpu": dist_ver("torch_tpu"),
            "libtpu": dist_ver("libtpu") or dist_ver("libtpu-nightly"),
            "transformers": dist_ver("transformers"),
        },
        "flint_checkpoint_inspection": {
            "architectures": cfg.get("architectures"),
            "model_type": cfg.get("model_type"),
            "auto_map": cfg.get("auto_map"),
            "shipped_wheels": shipped_artifacts,
            "num_hidden_layers": cfg.get("num_hidden_layers"),
            "num_experts": cfg.get("num_experts"),
            "num_experts_per_tok": cfg.get("num_experts_per_tok"),
            "sliding_window": cfg.get("sliding_window"),
        },
        "vllm_registry": {
            "total_registered_architectures": len(supported_archs),
            "flint_for_causal_lm_registered": has_flint_arch,
            "qwen3_moe_registered": "Qwen3MoeForCausalLM" in supported_archs,
        },
        "empirical_probes": {
            "model_impl_auto": p_auto,
            "model_impl_transformers": p_tf,
        },
        "architectural_comparison_vs_sglang_torchtpu": {
            "flint_model_support": {
                "sglang_torchtpu": "Native external model plugin (`flint_sglang-0.1.4`, `SGLANG_EXTERNAL_MODEL_PACKAGE=flint_sglang`) with `FlintForCausalLM` + `TpuPallasRpaV3AttnBackend` (`sinks=softmax_offset`) + hybrid SWA-513/Full-NoPE KV pools + 256-expert Pallas `router_topk` + `gmm_v2` Block-FP8.",
                "vllm_torchtpu": "`FlintForCausalLM` (`model_type='flint'`) is NOT in `vllm` ModelRegistry and checkpoint carries no HF `auto_map` (`flint` ships only `sglang/flint_sglang-0.1.4-py3-none-any.whl`), so both `--model-impl auto` and `--model-impl transformers` fail at config/architecture resolution.",
            },
            "tpu_multilora_tp_scaling": {
                "sglang_torchtpu": "Supports `TP=1, 2, 4, 8` on both `BF16` and `Block-FP8` base weights via `QKVParallelLinearWithLoRA` + `RowParallelLinearWithLoRA` + static-shape `TorchNativeLoRABackend` (`_tpu_weight_indices` buffer).",
                "vllm_torchtpu": "Punica `ColumnParallelLinearWithLoRA.slice_lora_a` raises `NotImplementedError` for `TP >= 2` on TPU, restricting vLLM TPU Multi-LoRA to `TP=1` (which fits <=27B models on 95 GiB TensorCores, or cannot initialize `count=1` PJRT slices without DP-attention).",
            },
        },
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2) + "\n")
    print(f"[vllm-probe] Saved {OUT_JSON}: auto={p_auto['status']} transformers={p_tf['status']}", flush=True)


if __name__ == "__main__":
    main()
