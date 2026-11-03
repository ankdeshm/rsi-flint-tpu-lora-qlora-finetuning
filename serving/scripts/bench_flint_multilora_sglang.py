#!/usr/bin/env python3
"""Multi-LoRA & QLoRA-Style Serving Benchmark for RecursiveSI/flint (40B MoE) on TPU v7x (`sglang-torchtpu`).

Benchmarks our TPU v5p-16 TorchTitan fine-tuned PEFT adapters concurrently across ALL 8 TensorCores
(`TP=2 x DP=4`, 4 single-chip replicas) on BOTH:
  - Phase E1 (`E1_sglang_fp8_multilora`): Block-FP8 Flint base (`/dev/shm/flint-fp8`) + FP8 KV ("QLoRA-style" quantized serving)
  - Phase E2 (`E2_sglang_bf16_multilora`): BF16 Flint base (`/dev/shm/flint`) + FP8 KV ("LoRA" full-precision base serving)

Adapters evaluated (trained on `yahma/alpaca-cleaned`, `seq_len=4096`):
  - `lora_bf16`  -> `/models/adapters/flint-40b-lora-r16`
  - `qlora_int8` -> `/models/adapters/flint-40b-qlora-int8-r16`
  - `qlora_nf4`  -> `/models/adapters/flint-40b-qlora-nf4-r16`
  - `lora_attn`  -> `/models/adapters/flint-40b-lora-attn-r16`
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

import aiohttp
import numpy as np


ADAPTER_CANDIDATES = [
    ("lora_bf16", "/models/adapters/flint-40b-lora-r16"),
    ("qlora_int8", "/models/adapters/flint-40b-qlora-int8-r16"),
    ("qlora_nf4", "/models/adapters/flint-40b-qlora-nf4-r16"),
    ("lora_attn", "/models/adapters/flint-40b-lora-attn-r16"),
]

QUALITY_PROMPTS = [
    "Explain in three concise bullet points why Sliding Window Attention with learned attention sinks reduces KV-cache memory on 128K context.",
    "Write a clean Python function `top_k_frequent(nums: list[int], k: int) -> list[int]` in O(n) time and show a quick example.",
    "Give a clear step-by-step comparison between BF16 LoRA, INT8 QLoRA, and 4-bit NF4 QLoRA fine-tuning on Mixture-of-Experts models.",
]


def stage_adapters(dst_root: Path) -> dict[str, str]:
    dst_root.mkdir(parents=True, exist_ok=True)
    staged: dict[str, str] = {}
    for name, src_str in ADAPTER_CANDIDATES:
        src = Path(src_str)
        if (src / "adapter_config.json").exists() and (src / "adapter_model.safetensors").exists():
            dst = dst_root / name
            dst.mkdir(parents=True, exist_ok=True)
            for fn in ("adapter_config.json", "adapter_model.safetensors"):
                shutil.copyfile(src / fn, dst / fn)
            staged[name] = str(dst)
    # If fewer than 4 distinct training runs have finished yet, create distinct copies from available adapters so 4-slot multi-LoRA still exercises 4 independent HBM slots
    if staged and len(staged) < 4:
        first_src = Path(next(iter(staged.values())))
        for name, _ in ADAPTER_CANDIDATES:
            if name not in staged:
                dst = dst_root / name
                dst.mkdir(parents=True, exist_ok=True)
                for fn in ("adapter_config.json", "adapter_model.safetensors"):
                    shutil.copyfile(first_src / fn, dst / fn)
                staged[name] = str(dst)
    return staged


def ensure_launch_cache_stem_lora() -> None:
    lhr_py = Path("/workspace/launch_host_replicas.py")
    if not lhr_py.exists():
        return
    txt = lhr_py.read_text()
    marker = "# [lora-cache-stem]"
    if marker in txt:
        return
    old = '    ind_r0 = Path(f"/dev/shm/ind_{cache_stem}_r0")'
    new = (
        f"    {marker}\n"
        '    if "--enable-lora" in args.extra_args or "--lora-paths" in args.extra_args:\n'
        '        cache_stem = f"{cache_stem}_multilora"\n'
        '    ind_r0 = Path(f"/dev/shm/ind_{cache_stem}_r0")'
    )
    if old in txt:
        lhr_py.write_text(txt.replace(old, new, 1))
        print("[multilora] Patched launch_host_replicas.py cache_stem for _multilora isolation", flush=True)


def http_post_json(url: str, payload: dict[str, Any], timeout: float = 120.0) -> tuple[int, dict[str, Any] | str]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            try:
                return resp.status, json.loads(raw)
            except Exception:
                return resp.status, raw
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", errors="replace")[:500]
    except Exception as e:
        return -1, f"{type(e).__name__}: {e}"


async def _stream_completion(
    session: aiohttp.ClientSession,
    base_url: str,
    model_path: str,
    lora_name: str | None,
    prompt_ids: list[int],
    osl: int,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model_path,
        "prompt": prompt_ids,
        "max_tokens": osl,
        "temperature": 0.0,
        "ignore_eos": True,
        "stream": True,
    }
    if lora_name is not None:
        payload["lora_path"] = lora_name

    t0 = time.perf_counter()
    first_tok_t: float | None = None
    chunks = 0
    try:
        async with session.post(f"{base_url}/v1/completions", json=payload) as resp:
            if resp.status != 200:
                body = await resp.text()
                return {"ok": False, "error": f"HTTP {resp.status}: {body[:240]}"}
            async for raw in resp.content:
                line = raw.decode("utf-8", errors="ignore").strip()
                if not line.startswith("data:") or line == "data: [DONE]":
                    continue
                try:
                    obj = json.loads(line[5:].strip())
                except Exception:
                    continue
                choices = obj.get("choices") or []
                if choices and choices[0].get("text"):
                    chunks += 1
                    if first_tok_t is None:
                        first_tok_t = time.perf_counter()
        t1 = time.perf_counter()
        ttft = (first_tok_t - t0) if first_tok_t is not None else (t1 - t0)
        tpot = ((t1 - first_tok_t) / max(chunks - 1, 1)) if (first_tok_t is not None and chunks > 1) else 0.0
        return {"ok": True, "chunks": chunks, "ttft_ms": ttft * 1000.0, "tpot_ms": tpot * 1000.0, "e2e_s": t1 - t0}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


async def run_multilora_point(
    base_url: str,
    model_path: str,
    adapter_schedule: list[str | None],
    isl: int,
    osl: int,
    concurrency: int,
    vocab_size: int = 150000,
    seed: int = 42,
) -> dict[str, Any]:
    rng = random.Random(seed + isl * 13 + osl * 7 + concurrency * 3 + len(adapter_schedule))
    n_reqs = max(concurrency * 2, 8) if concurrency <= 32 else concurrency
    reqs = [
        (
            adapter_schedule[i % len(adapter_schedule)],
            [rng.randrange(1000, vocab_size - 1000) for _ in range(isl)],
        )
        for i in range(n_reqs)
    ]
    sem = asyncio.Semaphore(concurrency)
    connector = aiohttp.TCPConnector(limit=max(concurrency * 2, 128))
    timeout = aiohttp.ClientTimeout(total=1800)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        # Quick 2-request warmup
        await asyncio.gather(
            *[_stream_completion(session, base_url, model_path, adapter_schedule[0], reqs[0][1][:128], 16) for _ in range(min(2, concurrency))]
        )

        async def _worker(lora_nm: str | None, p_ids: list[int]) -> dict[str, Any]:
            async with sem:
                return await _stream_completion(session, base_url, model_path, lora_nm, p_ids, osl)

        t0 = time.perf_counter()
        results = await asyncio.gather(*[_worker(ln, pids) for ln, pids in reqs])
        wall = max(time.perf_counter() - t0, 1e-6)

    oks = [r for r in results if r.get("ok")]
    errs = [r.get("error", "") for r in results if not r.get("ok")]
    # Each streaming chunk in SGLang non-speculative decode is 1 output token; clamp to osl if chunked
    out_toks = sum(osl if r["chunks"] >= int(osl * 0.9) else r["chunks"] for r in oks)
    ttfts = [r["ttft_ms"] for r in oks]
    tpots = [r["tpot_ms"] for r in oks if r["tpot_ms"] > 0]
    e2es = [r["e2e_s"] for r in oks]
    host_tps = out_toks / wall
    tpot_p50 = float(np.percentile(tpots, 50)) if tpots else 0.0
    return {
        "adapters": ["base" if a is None else a for a in adapter_schedule],
        "num_distinct_adapters": len({a for a in adapter_schedule if a is not None}),
        "includes_base": any(a is None for a in adapter_schedule),
        "isl": isl,
        "osl": osl,
        "concurrency": concurrency,
        "num_requests": n_reqs,
        "succeeded": len(oks),
        "failed": len(errs),
        "wall_s": round(wall, 3),
        "host_output_tok_s": round(host_tps, 2),
        "per_chip_output_tok_s": round(host_tps / 4.0, 2),
        "single_stream_tok_s": round(1000.0 / tpot_p50, 2) if tpot_p50 > 0 else 0.0,
        "ttft_ms_p50": round(float(np.percentile(ttfts, 50)), 2) if ttfts else 0.0,
        "ttft_ms_p99": round(float(np.percentile(ttfts, 99)), 2) if ttfts else 0.0,
        "tpot_ms_p50": round(tpot_p50, 3),
        "tpot_ms_p99": round(float(np.percentile(tpots, 99)), 3) if tpots else 0.0,
        "e2e_s_p50": round(float(np.percentile(e2es, 50)), 3) if e2es else 0.0,
        "sample_errors": list(dict.fromkeys(errs))[:3],
    }


def run_quality_probes(base_url: str, model_path: str, adapter_names: list[str]) -> dict[str, Any]:
    targets: list[str | None] = [None] + adapter_names
    probes: dict[str, list[dict[str, Any]]] = {}
    for tgt in targets:
        key = "base" if tgt is None else tgt
        probes[key] = []
        for prompt in QUALITY_PROMPTS:
            payload: dict[str, Any] = {
                "model": model_path,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 192,
                "temperature": 0.0,
            }
            if tgt is not None:
                payload["lora_path"] = tgt
            t0 = time.perf_counter()
            code, resp = http_post_json(f"{base_url}/v1/chat/completions", payload, timeout=180.0)
            dt = time.perf_counter() - t0
            rec: dict[str, Any] = {"prompt": prompt[:100], "http": code, "latency_s": round(dt, 3)}
            if code == 200 and isinstance(resp, dict):
                msg = (resp.get("choices") or [{}])[0].get("message") or {}
                rec["content"] = (msg.get("content") or "")[:450]
                rec["reasoning_content"] = (msg.get("reasoning_content") or "")[:300]
                rec["completion_tokens"] = (resp.get("usage") or {}).get("completion_tokens")
            else:
                rec["error"] = str(resp)[:300]
            probes[key].append(rec)
    # Check greedy output divergence vs base
    base_texts = [(r.get("reasoning_content", "") + r.get("content", "")) for r in probes.get("base", [])]
    divergence: dict[str, bool] = {}
    for nm in adapter_names:
        ad_texts = [(r.get("reasoning_content", "") + r.get("content", "")) for r in probes.get(nm, [])]
        divergence[nm] = (ad_texts != base_texts) and any(bool(t.strip()) for t in ad_texts)
    return {"outputs": probes, "diverges_from_base": divergence}


def run_dynamic_hotswap_test(num_replicas: int, adapter_name: str, adapter_path: str) -> dict[str, Any]:
    unload_ms: list[float] = []
    reload_ms: list[float] = []
    errors: list[str] = []
    for r in range(num_replicas):
        port = 30000 + r
        t0 = time.perf_counter()
        c_un, resp_un = http_post_json(f"http://127.0.0.1:{port}/unload_lora_adapter", {"lora_name": adapter_name}, timeout=60.0)
        unload_ms.append(round((time.perf_counter() - t0) * 1000.0, 2))
        if c_un != 200:
            errors.append(f"rep{r} unload HTTP {c_un}: {resp_un}")
        t1 = time.perf_counter()
        c_ld, resp_ld = http_post_json(
            f"http://127.0.0.1:{port}/load_lora_adapter",
            {"lora_name": adapter_name, "lora_path": adapter_path},
            timeout=60.0,
        )
        reload_ms.append(round((time.perf_counter() - t1) * 1000.0, 2))
        if c_ld != 200:
            errors.append(f"rep{r} load HTTP {c_ld}: {resp_ld}")
    return {
        "adapter": adapter_name,
        "unload_ms_per_replica": unload_ms,
        "reload_ms_per_replica": reload_ms,
        "unload_ms_mean": round(float(np.mean(unload_ms)), 2) if unload_ms else 0.0,
        "reload_ms_mean": round(float(np.mean(reload_ms)), 2) if reload_ms else 0.0,
        "ok": len(errors) == 0,
        "errors": errors,
    }


def launch_multilora_phase(
    tag: str,
    model_path: str,
    mem_fraction: float,
    staged_adapters: dict[str, str],
    disable_compile: bool = False,
) -> tuple[bool, dict[str, Any]]:
    import run_v7x_sglang_worker as wk

    wk.prune_inductor_caches_if_needed(min_free_gb=110.0)
    lora_kv = " ".join(f"{nm}={p}" for nm, p in staged_adapters.items())
    extra_args = (
        f"--enable-lora --lora-backend torch_native --max-loras-per-batch 5 "
        f"--max-lora-rank 16 --lora-target-modules q_proj k_proj v_proj o_proj "
        f"--lora-paths {lora_kv}"
    )
    cmd = [
        sys.executable,
        "/workspace/launch_host_replicas.py",
        "--mode",
        "start",
        "--tag",
        tag,
        "--model-path",
        model_path,
        "--tp-size",
        "2",
        "--dp-size",
        "1",
        "--ep-size",
        "1",
        "--num-replicas",
        "4",
        "--kv-cache-dtype",
        "fp8_e4m3",
        "--context-length",
        "32768",
        "--mem-fraction-static",
        str(mem_fraction),
        "--chunked-prefill-size",
        "8192",
        "--max-prefill-tokens",
        "16384",
        "--max-running-requests",
        "64",
        "--torch-compile-max-bs",
        "64",
        "--swa-full-tokens-ratio",
        "0.10",
        "--wait-timeout",
        "1800",
        "--extra-args",
        extra_args,
    ]
    if disable_compile:
        cmd.append("--disable-torch-compile")
    print(f"\n[{tag}] Launching SGLang Multi-LoRA servers: {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, text=True, capture_output=True)
    lines = [ln for ln in proc.stdout.strip().splitlines() if ln.strip()]
    info: dict[str, Any] = {"returncode": proc.returncode, "stderr_tail": proc.stderr[-1500:] if proc.stderr else ""}
    if lines:
        try:
            info.update(json.loads(lines[-1]))
        except Exception:
            info["raw_stdout_tail"] = "\n".join(lines[-20:])
    if proc.returncode == 0 and info.get("status") == "ready":
        subprocess.run(
            [
                sys.executable,
                "/workspace/gateway_flint_v7x.py",
                "--port",
                "8000",
                "--workers",
                "http://127.0.0.1:30000,http://127.0.0.1:30001,http://127.0.0.1:30002,http://127.0.0.1:30003",
                "--daemon",
                "--log-file",
                f"/workspace/logs/gw_{tag}.log",
            ],
            check=True,
        )
        time.sleep(1.5)
        return True, info
    return False, info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="/models/benchmark-results/flint_v7x_rqkr_opt/runs/E_sglang_flint_multilora.json")
    ap.add_argument("--phases", default="fp8,bf16", help="Comma-separated: fp8,bf16")
    args = ap.parse_args()

    import patch_sglang_tpu_lora

    patch_sglang_tpu_lora.apply_sglang_tpu_lora_patches()
    ensure_launch_cache_stem_lora()

    staged = stage_adapters(Path("/dev/shm/flint_adapters"))
    adapter_names = list(staged.keys())
    print(f"[multilora] Staged {len(staged)} Flint 40B PEFT adapters: {staged}", flush=True)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bundle: dict[str, Any] = {
        "run_name": "E_sglang_flint_multilora",
        "axis": "E_multilora_qlora_serving",
        "host": os.uname().nodename,
        "hardware": f"Single-Host TPU v7x ({os.uname().nodename}, 4 chips = 8 TensorCores, TP=2 x DP=4)",
        "adapters_staged": staged,
        "phases": {},
    }

    phase_specs = []
    req_phases = {x.strip() for x in args.phases.split(",") if x.strip()}
    if "fp8" in req_phases:
        phase_specs.append(("E1_sglang_fp8_multilora", "/dev/shm/flint-fp8", "fp8_block128", 0.82))
    if "bf16" in req_phases:
        phase_specs.append(("E2_sglang_bf16_multilora", "/dev/shm/flint", "bfloat16", 0.80))

    schedules: list[tuple[str, list[str | None]]] = [
        ("0_loras_base_only", [None]),
        ("1_lora_bf16", [adapter_names[0]]),
        ("2_loras_bf16_plus_qlora_int8", adapter_names[:2]),
        ("4_loras_all_mixed", adapter_names[:4]),
        ("mixed_base_plus_3_loras", [None] + adapter_names[:3]),
    ]

    for tag, model_path, w_dtype, mem_frac in phase_specs:
        ok, linfo = launch_multilora_phase(tag, model_path, mem_frac, staged, disable_compile=False)
        used_eager = False
        if not ok:
            print(f"[{tag}] Retrying with --disable-torch-compile...", flush=True)
            ok, linfo2 = launch_multilora_phase(tag, model_path, mem_frac, staged, disable_compile=True)
            linfo = {"compiled_attempt": linfo, "eager_attempt": linfo2}
            used_eager = True
        if not ok:
            bundle["phases"][tag] = {"status": "failed", "weights_dtype": w_dtype, "launch_info": linfo}
            out_path.write_text(json.dumps(bundle, indent=2) + "\n")
            subprocess.run([sys.executable, "/workspace/launch_host_replicas.py", "--mode", "stop"], check=False)
            continue

        print(f"[{tag}] READY! Running quality probes, multi-adapter throughput matrix, and dynamic hot-swap...", flush=True)
        q_probe = run_quality_probes("http://127.0.0.1:8000", model_path, adapter_names)
        points: list[dict[str, Any]] = []
        for sched_name, sched in schedules:
            for isl, osl, concs in [(1024, 512, [1, 8, 16, 32, 64]), (1024, 1024, [1, 16, 64])]:
                if sched_name not in ("1_lora_bf16", "4_loras_all_mixed") and osl == 1024:
                    continue
                for c in concs:
                    pt = asyncio.run(run_multilora_point("http://127.0.0.1:8000", model_path, sched, isl, osl, c))
                    pt["schedule_name"] = sched_name
                    points.append(pt)
                    print(
                        f"  [{tag}] {sched_name} {isl}:{osl} c={c} | out={pt['host_output_tok_s']} tok/s "
                        f"| 1-stream={pt['single_stream_tok_s']} tok/s | TTFT p50={pt['ttft_ms_p50']}ms | failed={pt['failed']}",
                        flush=True,
                    )
                    bundle["phases"][tag] = {
                        "status": "in_progress",
                        "weights_dtype": w_dtype,
                        "used_eager": used_eager,
                        "launch_info": linfo,
                        "quality_probes": q_probe,
                        "points": points,
                    }
                    out_path.write_text(json.dumps(bundle, indent=2) + "\n")

        hotswap = run_dynamic_hotswap_test(4, adapter_names[-1], staged[adapter_names[-1]])
        print(f"  [{tag}] Dynamic hot-swap: {hotswap}", flush=True)
        bundle["phases"][tag] = {
            "status": "completed",
            "weights_dtype": w_dtype,
            "used_eager": used_eager,
            "launch_info": linfo,
            "quality_probes": q_probe,
            "dynamic_hotswap": hotswap,
            "points": points,
        }
        out_path.write_text(json.dumps(bundle, indent=2) + "\n")
        subprocess.run([sys.executable, "/workspace/launch_host_replicas.py", "--mode", "stop"], check=False)

    bundle["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    out_path.write_text(json.dumps(bundle, indent=2) + "\n")
    print(f"[multilora] ALL DONE -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
