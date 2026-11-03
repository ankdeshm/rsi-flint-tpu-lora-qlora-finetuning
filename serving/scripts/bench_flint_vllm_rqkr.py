#!/usr/bin/env python3
"""Empirical vllm-torchtpu vs sglang-torchtpu Engine Comparison, Multi-LoRA & N-Gram SpecDec Probe for RecursiveSI/flint on TPU v7x.

Executes two stages on a single TPU v7x host (4 dual-core chips = 8 TensorCores):
  Stage 1 (Stock Upstream `vllm-torchtpu:v0.32.0`):
    - Verifies `ModelRegistry.get_supported_archs()` (`FlintForCausalLM` absent) and probes `--model-impl auto` + `--model-impl transformers`.
  Stage 2 (Plugin-Enabled `vllm-torchtpu` via `install_flint_vllm.py`, isolated single-host `TORCH_TPU_TOPOLOGY=2,2,1`):
    - `vllm_fp8_tp1_dp8_ep8_multilora`: `TP=1 x DP=8 x EP=8 --enable-lora` serving our TorchTitan-exported PEFT adapters (`0`, `1`, `4` concurrent LoRAs).
    - `vllm_fp8_tp2_dp4_ep8_lora_probe`: `TP=2 x DP=4 x EP=8 --enable-lora` (verifies `TP >= 2` LoRA constraint on TPU).
    - `vllm_fp8_tp2_dp4_ep8_ngram5`: `TP=2 x DP=4 x EP=8` + `ngram` speculative decoding (`num_speculative_tokens=5`) on structured Python/JSON code-editing & standard prompts.
"""
from __future__ import annotations

import concurrent.futures as cf
import importlib.metadata
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
from typing import Any
import urllib.error
import urllib.request

import numpy as np


MODEL_BF16_GCS = Path("/models/models/RecursiveSI/flint")
MODEL_FP8_GCS = Path("/models/models/RecursiveSI/flint-fp8")
OUT_JSON = Path("/models/benchmark-results/flint_v7x_rqkr_opt/runs/E3_vllm_torchtpu_flint_comparison.json")

ADAPTERS = [
    ("lora_bf16", "/models/adapters/flint-40b-lora-r16"),
    ("qlora_int8", "/models/adapters/flint-40b-qlora-int8-r16"),
    ("qlora_nf4", "/models/adapters/flint-40b-qlora-nf4-r16"),
    ("lora_attn", "/models/adapters/flint-40b-lora-attn-r16"),
]

CODE_EDIT_PROMPT = (
    "Refactor the following Python module so every method adds type annotations, docstrings, and logging while "
    "preserving exact function names, field names, and return dictionaries:\n\n"
    "```python\n"
    "class OrderRouter:\n"
    "    def __init__(self, venues, max_slippage_bps=12, timeout_ms=250, enable_hedging=True):\n"
    "        self.venues = venues\n"
    "        self.max_slippage_bps = max_slippage_bps\n"
    "        self.timeout_ms = timeout_ms\n"
    "        self.enable_hedging = enable_hedging\n"
    "        self.metrics = {'routed': 0, 'filled': 0, 'rejected': 0, 'hedged': 0}\n\n"
    "    def route_order(self, order_id, symbol, side, quantity, limit_price):\n"
    "        best_venue = None\n"
    "        best_px = None\n"
    "        for venue in self.venues:\n"
    "            quote = venue.get_quote(symbol, side, quantity)\n"
    "            if quote is None or not quote.get('available', False):\n"
    "                continue\n"
    "            if best_px is None or (side == 'BUY' and quote['price'] < best_px) or (side == 'SELL' and quote['price'] > best_px):\n"
    "                best_px = quote['price']\n"
    "                best_venue = venue\n"
    "        if best_venue is None:\n"
    "            self.metrics['rejected'] += 1\n"
    "            return {'order_id': order_id, 'status': 'REJECTED', 'venue': None, 'fill_price': None}\n"
    "        self.metrics['routed'] += 1\n"
    "        fill = best_venue.execute(order_id, symbol, side, quantity, limit_price, self.timeout_ms)\n"
    "        self.metrics['filled'] += int(fill['status'] == 'FILLED')\n"
    "        return {'order_id': order_id, 'status': fill['status'], 'venue': best_venue.name, 'fill_price': fill['price']}\n"
    "```\nOutput the complete refactored Python module:\n```python\n"
)


def dist_ver(pkg: str) -> str | None:
    try:
        return importlib.metadata.version(pkg)
    except Exception:
        return None


def http_ok(port: int, path: str = "/health") -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=2.5) as r:
            return r.status == 200
    except Exception:
        return False


def stage_dir(src: Path, dst: Path) -> float:
    dst.mkdir(parents=True, exist_ok=True)
    files = [p for p in src.rglob("*") if p.is_file() and ".cache" not in p.parts]
    t0 = time.time()

    def _cp(p: Path) -> int:
        out = dst / p.relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        if not out.exists() or out.stat().st_size != p.stat().st_size:
            shutil.copyfile(p, out)
        return p.stat().st_size

    with cf.ThreadPoolExecutor(max_workers=16) as ex:
        tot = sum(ex.map(_cp, files))
    dt = max(time.time() - t0, 1e-3)
    print(f"[vllm-rqkr] Staged {tot/1e9:.2f} GB from {src} -> {dst} in {dt:.1f}s", flush=True)
    return round(dt, 2)


def probe_stock_vllm_launch(tag: str, model_dir: str, extra_flags: list[str], port: int = 8100, timeout_s: int = 120) -> dict[str, Any]:
    log_path = Path(f"/tmp/vllm_stock_{tag}.log")
    if log_path.exists():
        log_path.unlink()
    env = os.environ.copy()
    env["PJRT_DEVICE"] = "TPU"
    env["TPU_SKIP_MDS_QUERY"] = "1"
    env["TORCH_TPU_TOPOLOGY"] = "2,2,1"
    cmd = [
        "vllm",
        "serve",
        model_dir,
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
        "log_tail": "\n".join(lines[-40:]),
    }


def build_single_host_ray_env() -> dict[str, str]:
    env = os.environ.copy()
    for k in (
        "TPU_PROCESS_ADDRESSES",
        "TPU_WORKER_HOSTNAMES",
        "TPU_WORKER_ID",
        "TPU_CHIPS_PER_HOST_BOUNDS",
        "TPU_HOST_BOUNDS",
        "TPU_TOPOLOGY",
        "TPU_TOPOLOGY_ALT",
        "TPU_TOPOLOGY_WRAP",
        "TORCH_DEVICE_BACKEND_AUTOLOAD",
        "TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL",
    ):
        env.pop(k, None)
    copy_keys = [
        "TPU_MULTIHOST_BACKEND",
        "TPU_SKIP_MDS_QUERY",
        "TORCH_TPU_TOPOLOGY",
        "RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS",
        "VLLM_PLUGINS",
        "VLLM_TARGET_DEVICE",
        "PJRT_DEVICE",
        "HF_HOME",
        "TMPDIR",
        "VLLM_CACHE_ROOT",
        "VLLM_XLA_CACHE_PATH",
        "TORCHINDUCTOR_CACHE_DIR",
        "TORCHINDUCTOR_AUTOGRAD_CACHE",
        "TORCH_DIST_TIMEOUT",
    ]
    for d in ("/tmp/hf_cache", "/tmp/vllm_tmp", "/tmp/vllm_cache", "/tmp/torchinductor"):
        Path(d).mkdir(parents=True, exist_ok=True)
    env.update(
        {
            "TPU_MULTIHOST_BACKEND": "ray",
            "TPU_PROCESS_ADDRESSES": "",
            "TPU_WORKER_HOSTNAMES": "",
            "TPU_SKIP_MDS_QUERY": "1",
            "TORCH_TPU_TOPOLOGY": "2,2,1,2",
            "RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS": "1",
            "VLLM_PLUGINS": "torchtpu",
            "VLLM_TARGET_DEVICE": "tpu",
            "PJRT_DEVICE": "TPU",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "VLLM_NO_USAGE_STATS": "1",
            "HF_HOME": "/tmp/hf_cache",
            "TMPDIR": "/tmp/vllm_tmp",
            "VLLM_CACHE_ROOT": "/tmp/vllm_cache",
            "VLLM_XLA_CACHE_PATH": "/tmp/vllm_cache",
            "TORCHINDUCTOR_CACHE_DIR": "/tmp/torchinductor",
            "TORCHINDUCTOR_AUTOGRAD_CACHE": "0",
            "TORCH_DIST_TIMEOUT": "1800",
            "VLLM_RPC_TIMEOUT": "1800000",
            "VLLM_SHM_BROADCAST_TIMEOUT_S": "1800",
            "VLLM_ENGINE_ITERATION_TIMEOUT_S": "1800",
            "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS": "1800",
            "VLLM_ENGINE_READY_TIMEOUT_S": "2400",
            "VLLM_RAY_EXTRA_ENV_VARS_TO_COPY": ",".join(copy_keys),
        }
    )
    return env


def clean_vllm_ray(env: dict[str, str]) -> None:
    subprocess.run(["ray", "stop", "--force"], env=env, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    kill_sh = (
        "for p in $(pgrep -f 'vllm serve|VLLM::|EngineCore|ray::|gcs_server|raylet'); do "
        "[ \"$p\" != \"$$\" ] && kill -9 \"$p\" 2>/dev/null || true; done"
    )
    subprocess.run(["bash", "-lc", kill_sh], check=False)
    time.sleep(2.5)


def sse_one(port: int, model_name: str, prompt: str | list[int], max_tokens: int) -> dict[str, Any]:
    url = f"http://127.0.0.1:{port}/v1/completions"
    body = json.dumps(
        {
            "model": model_name,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "min_tokens": max_tokens,
            "ignore_eos": True,
            "temperature": 0.0,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    ).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST", headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    t_first: float | None = None
    comp_toks = 0
    chunks = 0
    try:
        with urllib.request.urlopen(req, timeout=600.0) as resp:
            for raw in resp:
                now = time.perf_counter()
                ln = raw.decode("utf-8", errors="ignore").strip()
                if not ln.startswith("data:"):
                    continue
                d = ln[5:].strip()
                if d == "[DONE]":
                    break
                try:
                    ev = json.loads(d)
                except Exception:
                    continue
                chs = ev.get("choices") or []
                if chs and chs[0].get("text"):
                    if t_first is None:
                        t_first = now
                    chunks += 1
                u = ev.get("usage")
                if u and u.get("completion_tokens"):
                    comp_toks = int(u["completion_tokens"])
        t1 = time.perf_counter()
        if t_first is None:
            t_first = t1
        n_out = comp_toks if comp_toks > 0 else max(chunks, 1)
        tpot_ms = ((t1 - t_first) * 1000.0) / max(n_out - 1, 1)
        return {"ok": True, "out": n_out, "ttft_ms": (t_first - t0) * 1000.0, "tpot_ms": tpot_ms, "e2e_s": t1 - t0}
    except Exception as e:
        return {"ok": False, "error": str(e)[:300]}


_TOK = None


def _make_prompt_str(rng: random.Random, isl: int) -> str:
    global _TOK
    if _TOK is None:
        tok_f = Path("/dev/shm/flint-fp8/tokenizer.json")
        if not tok_f.exists():
            tok_f = MODEL_BF16_GCS / "tokenizer.json"
        if tok_f.exists():
            from tokenizers import Tokenizer

            _TOK = Tokenizer.from_file(str(tok_f))
    ids = [rng.randint(500, 30000) for _ in range(isl)]
    if _TOK is not None:
        return _TOK.decode(ids)
    return " ".join(f"tok{x}" for x in ids[: max(isl // 2, 64)])


def _save_report(report: dict[str, Any]) -> None:
    raw = (json.dumps(report, indent=2) + "\n").encode("utf-8")
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_bytes(raw)
    try:
        mreq = urllib.request.Request(
            "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token",
            headers={"Metadata-Flavor": "Google"},
        )
        with urllib.request.urlopen(mreq, timeout=5.0) as mr:
            tok = json.loads(mr.read().decode("utf-8"))["access_token"]
        bucket = os.environ.get("GCS_ARTIFACTS_BUCKET", "")
        if bucket:
            obj_name = "benchmark-results/flint_v7x_rqkr_opt/runs/E3_vllm_torchtpu_flint_comparison.json"
            up_url = f"https://storage.googleapis.com/upload/storage/v1/b/{bucket}/o?uploadType=media&name={urllib.parse.quote(obj_name, safe='')}"
            ureq = urllib.request.Request(
                up_url,
                data=raw,
                method="POST",
                headers={"Authorization": f"Bearer {tok}", "Content-Type": "application/json"},
            )
            with urllib.request.urlopen(ureq, timeout=15.0) as ur:
                ur.read()
    except Exception as exc:
        print(f"[vllm-rqkr] Direct GCS upload note: {exc}", flush=True)


def bench_vllm_point(port: int, models_schedule: list[str], isl: int, osl: int, conc: int, coding: bool = False) -> dict[str, Any]:
    rng = random.Random(1337 + isl + conc)
    n_reqs = 4 if conc == 1 else min(max(conc * 2, 8), 64)
    prompts: list[str] = []
    for i in range(n_reqs):
        if coding:
            prompts.append(f"{CODE_EDIT_PROMPT}# req_id={i}\n")
        else:
            prompts.append(_make_prompt_str(rng, isl))
    # Warm 1 request
    sse_one(port, models_schedule[0], prompts[0], 8)
    tw0 = time.perf_counter()
    with cf.ThreadPoolExecutor(max_workers=conc) as ex:
        futs = [ex.submit(sse_one, port, models_schedule[i % len(models_schedule)], prompts[i], osl) for i in range(n_reqs)]
        res = [f.result() for f in futs]
    wall = max(time.perf_counter() - tw0, 1e-6)
    oks = [r for r in res if r.get("ok")]
    errs = [r for r in res if not r.get("ok")]
    tot_out = sum(r["out"] for r in oks)
    ttfts = [r["ttft_ms"] for r in oks]
    tpots = [r["tpot_ms"] for r in oks]
    tpot_p50 = float(np.percentile(tpots, 50)) if tpots else 0.0
    return {
        "workload": "coding_edit" if coding else f"{isl}:{osl}",
        "models_schedule": models_schedule,
        "concurrency": conc,
        "n_reqs": n_reqs,
        "ok": len(oks),
        "failed": len(errs),
        "host_output_tok_s": round(tot_out / wall, 2),
        "single_stream_tok_s": round(1000.0 / tpot_p50, 2) if tpot_p50 > 0 else 0.0,
        "ttft_ms_p50": round(float(np.percentile(ttfts, 50)), 2) if ttfts else 0.0,
        "tpot_ms_p50": round(tpot_p50, 3),
        "sample_error": errs[0]["error"] if errs else None,
    }


def run_plugin_experiment(name: str, model_dir: str, tp: int, dp: int, extra_args: list[str], env: dict[str, str], timeout_s: int = 1200) -> tuple[bool, subprocess.Popen[Any] | None, dict[str, Any]]:
    clean_vllm_ray(env)
    subprocess.run(["ray", "start", "--head", "--port=6379", "--labels=ray.io/tpu-worker-id=0"], env=env, check=True)
    log_path = Path(f"/tmp/{name}.log")
    comp_cfg = json.dumps({"compile_sizes": [4, 16, 64, 128, 256, 512, 1024, 2048]})
    use_async = not any("ngram" in a for a in extra_args)
    cmd = [
        "vllm",
        "serve",
        model_dir,
        f"--tokenizer={model_dir}",
        "--served-model-name=RecursiveSI/flint",
        "--host=0.0.0.0",
        "--port=8110",
        "--api-server-count=1",
        f"--tensor-parallel-size={tp}",
        f"--data-parallel-size={dp}",
        "--safetensors-load-strategy=prefetch",
        "--kv-cache-dtype=fp8",
        "--gpu-memory-utilization=0.84",
        "--max-model-len=16384",
        "--max-num-seqs=64",
        "--max-num-batched-tokens=2048",
        f"--compilation-config={comp_cfg}",
        "--enable-prefix-caching",
        "--async-scheduling" if use_async else "--no-async-scheduling",
        "--trust-remote-code",
        *extra_args,
    ]
    print(f"[vllm-rqkr] Launching {name}: {' '.join(cmd)}", flush=True)
    t0 = time.time()
    lf = log_path.open("w")
    proc = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
    while time.time() - t0 < timeout_s:
        if http_ok(8110, "/v1/models"):
            return True, proc, {"status": "ready", "boot_sec": round(time.time() - t0, 2), "command": " ".join(cmd)}
        if proc.poll() is not None:
            tail = log_path.read_text(errors="replace")[-3000:]
            clean_vllm_ray(env)
            return False, None, {"status": "exited", "returncode": proc.returncode, "boot_sec": round(time.time() - t0, 2), "log_tail": tail}
        txt = log_path.read_text(errors="replace") if log_path.exists() else ""
        if "NotImplementedError:" in txt and ("slice_lora_a" in txt or "LoRA" in txt or "Speculative" in txt):
            time.sleep(3.0)
            clean_vllm_ray(env)
            return False, None, {"status": "unsupported_exception", "boot_sec": round(time.time() - t0, 2), "log_tail": txt[-3000:]}
        time.sleep(3.0)
    tail = log_path.read_text(errors="replace")[-3000:] if log_path.exists() else ""
    clean_vllm_ray(env)
    return False, None, {"status": "timeout", "boot_sec": round(time.time() - t0, 2), "log_tail": tail}


def main() -> None:
    import urllib.parse  # noqa: F401

    shipped_artifacts = sorted(str(p.relative_to(MODEL_BF16_GCS)) for p in MODEL_BF16_GCS.rglob("*.whl")) if MODEL_BF16_GCS.exists() else []
    cfg = json.loads((MODEL_BF16_GCS / "config.json").read_text()) if (MODEL_BF16_GCS / "config.json").exists() else {}

    supported_archs: list[str] = []
    has_flint_arch = False
    try:
        from vllm.model_executor.models import ModelRegistry

        supported_archs = sorted(ModelRegistry.get_supported_archs())
        has_flint_arch = "FlintForCausalLM" in supported_archs
    except Exception as e:
        supported_archs = [f"error: {e}"]

    # Stage 1: Stock upstream vllm-torchtpu probes
    p_auto = probe_stock_vllm_launch("native_auto", str(MODEL_BF16_GCS), ["--model-impl", "auto"], port=8101, timeout_s=90)
    p_tf = probe_stock_vllm_launch("transformers_backend", str(MODEL_BF16_GCS), ["--model-impl", "transformers"], port=8102, timeout_s=90)

    report: dict[str, Any] = {
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
        "plugin_enabled_benchmarks": {},
    }
    _save_report(report)

    # Stage 2: Install custom FlintForCausalLM vLLM plugin if staged alongside this script
    installer = Path(__file__).resolve().parent / "install_flint_vllm.py"
    if installer.exists() and MODEL_FP8_GCS.exists():
        print("[vllm-rqkr] Installing custom FlintForCausalLM vllm-torchtpu plugin...", flush=True)
        subprocess.run([sys.executable, str(installer)], check=True)
        stage_dir(MODEL_FP8_GCS, Path("/dev/shm/flint-fp8"))
        staged_loras: list[tuple[str, str]] = []
        for nm, ap in ADAPTERS:
            src_a = Path(ap)
            if (src_a / "adapter_config.json").exists():
                dst_a = Path("/dev/shm/flint_adapters") / nm
                stage_dir(src_a, dst_a)
                staged_loras.append((nm, str(dst_a)))

        env = build_single_host_ray_env()

        # 2a. Probe TP=2 Multi-LoRA on vllm-torchtpu (expected Punica TP>=2 constraint or ready)
        if staged_loras:
            lora_flags = [
                "--enable-lora",
                "--max-loras=4",
                "--max-lora-rank=16",
                "--lora-modules",
                *[f"{nm}={p}" for nm, p in staged_loras],
            ]
            ok_tp2, proc_tp2, info_tp2 = run_plugin_experiment(
                "vllm_fp8_tp2_dp4_ep8_lora_probe",
                "/dev/shm/flint-fp8",
                tp=2,
                dp=4,
                extra_args=["--enable-expert-parallel", *lora_flags],
                env=env,
                timeout_s=900,
            )
            pts_tp2 = []
            if ok_tp2:
                for c in [1, 16, 64]:
                    pts_tp2.append(bench_vllm_point(8110, [nm for nm, _ in staged_loras], 1024, 512, c))
                clean_vllm_ray(env)
            print(f"[vllm-rqkr] vllm_fp8_tp2_dp4_ep8_lora_probe -> status={info_tp2.get('status')}", flush=True)
            report["plugin_enabled_benchmarks"]["vllm_fp8_tp2_dp4_ep8_lora_probe"] = {"launch": info_tp2, "points": pts_tp2}
            _save_report(report)

            # 2b. TP=1 x DP=8 x EP=8 Multi-LoRA serving on vllm-torchtpu
            ok_tp1, proc_tp1, info_tp1 = run_plugin_experiment(
                "vllm_fp8_tp1_dp8_ep8_multilora",
                "/dev/shm/flint-fp8",
                tp=1,
                dp=8,
                extra_args=["--enable-expert-parallel", *lora_flags],
                env=env,
                timeout_s=1200,
            )
            pts_tp1 = []
            if ok_tp1:
                for sched_label, sched_models in [
                    ("0_loras_base", ["RecursiveSI/flint"]),
                    ("1_lora_bf16", [staged_loras[0][0]]),
                    ("4_loras_mixed", [nm for nm, _ in staged_loras]),
                ]:
                    for c in [1, 16, 64]:
                        pt = bench_vllm_point(8110, sched_models, 1024, 512, c)
                        pt["schedule_name"] = sched_label
                        pts_tp1.append(pt)
                        print(f"  [vllm-multilora] {sched_label} c={c} -> {pt['host_output_tok_s']} tok/s | 1-stream={pt['single_stream_tok_s']} tok/s", flush=True)
                clean_vllm_ray(env)
            print(f"[vllm-rqkr] vllm_fp8_tp1_dp8_ep8_multilora -> status={info_tp1.get('status')} pts={len(pts_tp1)}", flush=True)
            report["plugin_enabled_benchmarks"]["vllm_fp8_tp1_dp8_ep8_multilora"] = {"launch": info_tp1, "points": pts_tp1}
            _save_report(report)

        # 2c. N-Gram Speculative Decoding on vllm-torchtpu (TP=4 x DP=2 x EP=8, --no-async-scheduling)
        ok_ng, proc_ng, info_ng = run_plugin_experiment(
            "vllm_fp8_tp4_dp2_ep8_ngram5",
            "/dev/shm/flint-fp8",
            tp=4,
            dp=2,
            extra_args=[
                "--enable-expert-parallel",
                '--speculative-config={"method":"ngram","num_speculative_tokens":5,"prompt_lookup_max":5,"prompt_lookup_min":2}',
            ],
            env=env,
            timeout_s=1200,
        )
        pts_ng = []
        if ok_ng:
            for c in [1, 4, 16]:
                pt_code = bench_vllm_point(8110, ["RecursiveSI/flint"], 1024, 384, c, coding=True)
                pts_ng.append(pt_code)
                print(f"  [vllm-ngram5] coding_edit c={c} -> {pt_code['host_output_tok_s']} tok/s | 1-stream={pt_code['single_stream_tok_s']} tok/s", flush=True)
            for c in [1, 16]:
                pt_std = bench_vllm_point(8110, ["RecursiveSI/flint"], 1024, 512, c, coding=False)
                pts_ng.append(pt_std)
                print(f"  [vllm-ngram5] standard 1024:512 c={c} -> {pt_std['host_output_tok_s']} tok/s | 1-stream={pt_std['single_stream_tok_s']} tok/s", flush=True)
            clean_vllm_ray(env)
        print(f"[vllm-rqkr] vllm_fp8_tp4_dp2_ep8_ngram5 -> status={info_ng.get('status')} pts={len(pts_ng)}", flush=True)
        report["plugin_enabled_benchmarks"]["vllm_fp8_tp4_dp2_ep8_ngram5"] = {"launch": info_ng, "points": pts_ng}

    report["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _save_report(report)
    print(f"[vllm-probe] ALL DONE -> {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
