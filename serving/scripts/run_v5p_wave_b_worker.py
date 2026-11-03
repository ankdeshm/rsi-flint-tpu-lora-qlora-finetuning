#!/usr/bin/env python3
"""Wave B 4-Host TPU v5p Serving Optimization & Multi-LoRA Benchmark Worker for RecursiveSI/flint (39.8B MoE).

Runs 4 independent 1-host (4-chip `ct5p-hightpu-4t`) tracks concurrently across `tpu-v5p-16-pool`:
  - Pod 0 (`JOB_COMPLETION_INDEX=0`): Block-FP8 Base + FP8 KV Multi-LoRA Serving (`TP=2 x DP=2` / `TP=4`)
  - Pod 1 (`JOB_COMPLETION_INDEX=1`): BF16 Base + FP8 KV Multi-LoRA Serving (`TP=2 x DP=2` / `TP=4`)
  - Pod 2 (`JOB_COMPLETION_INDEX=2`): N-Gram Speculative Decoding (`draft=4, 6`) vs Baseline (`TP=4`, FP8 & BF16)
  - Pod 3 (`JOB_COMPLETION_INDEX=3`): RadixCache Prefix-Hit (`0%/80%/95%` vs `--disable-radix-cache`) + `swa_ratio` (`0.05/0.10/0.20`) at `4K..128K`
"""
from __future__ import annotations

import asyncio
import concurrent.futures
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

import aiohttp
import numpy as np

RESULTS_DIR = Path("/models/benchmark-results/flint_v5p_wave_b")

ADAPTER_CANDIDATES = [
    ("lora_bf16", "/models/adapters/flint-40b-lora-r16"),
    ("qlora_int8", "/models/adapters/flint-40b-qlora-int8-r16"),
    ("qlora_nf4", "/models/adapters/flint-40b-qlora-nf4-r16"),
    ("lora_attn", "/models/adapters/flint-40b-lora-attn-r16"),
    ("lora_16k", "/models/adapters/flint-40b-lora-16k"),
]

QUALITY_PROMPTS = [
    "Explain in three concise bullet points why Sliding Window Attention with learned attention sinks reduces KV-cache memory on 128K context.",
    "Write a clean Python function `top_k_frequent(nums: list[int], k: int) -> list[int]` in O(n) time and show a quick example.",
    "Give a clear step-by-step comparison between BF16 LoRA, INT8 QLoRA, and 4-bit NF4 QLoRA fine-tuning on Mixture-of-Experts models.",
]

CODE_PREFIX_TEXT = (
    "import dataclasses\nfrom typing import Any\n\n"
    "class PagedRadixRouter:\n"
    "    def __init__(self, num_shards: int = 4, page_size: int = 64) -> None:\n"
    "        self.num_shards = num_shards\n"
    "        self.page_size = page_size\n"
    "        self.prefix_table: dict[tuple[int, ...], int] = {}\n\n"
    "    def route_prefix(self, token_ids: list[int]) -> int:\n"
    "        key = tuple(token_ids[: min(len(token_ids), 256)])\n"
    "        if key not in self.prefix_table:\n"
    "            self.prefix_table[key] = hash(key) % self.num_shards\n"
    "        return self.prefix_table[key]\n\n"
) * 64


def log(idx: int, msg: str) -> None:
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    print(f"[{ts}][v5p-waveB-pod{idx}] {msg}", flush=True)


def stage_tree(src_s: str, dst_s: str, idx: int) -> None:
    src = Path(src_s)
    dst = Path(dst_s)
    marker = dst / ".staged_ok"
    if marker.exists():
        return
    if not (src / "config.json").exists():
        return
    dst.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    files = [p for p in src.rglob("*") if p.is_file() and ".cache" not in p.parts]

    def copy_one(p: Path) -> int:
        rel = p.relative_to(src)
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists() and out.stat().st_size == p.stat().st_size:
            return p.stat().st_size
        shutil.copyfile(p, out)
        return p.stat().st_size

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex:
        total = sum(ex.map(copy_one, files))
    marker.write_text("ok\n")
    dt = max(time.time() - t0, 1e-3)
    log(idx, f"Staged {dst_s} ({total / 1e9:.2f} GB) in {dt:.1f}s ({total / 1e9 / dt:.2f} GB/s)")


def stage_adapters(dst_root: Path) -> dict[str, str]:
    dst_root.mkdir(parents=True, exist_ok=True)
    staged: dict[str, str] = {}
    for name, src_s in ADAPTER_CANDIDATES:
        src = Path(src_s)
        if (src / "adapter_config.json").exists() and (src / "adapter_model.safetensors").exists():
            dst = dst_root / name
            dst.mkdir(parents=True, exist_ok=True)
            for fn in ("adapter_config.json", "adapter_model.safetensors"):
                shutil.copyfile(src / fn, dst / fn)
            staged[name] = str(dst)
    return staged


def kill_sglang() -> None:
    subprocess.run(
        [
            "bash",
            "-c",
            "pkill -9 -f '[s]glang.launch_server' 2>/dev/null || true; "
            "pkill -9 -f '[s]glang::' 2>/dev/null || true; "
            "pkill -9 -f '[g]ateway_flint_v7x.py' 2>/dev/null || true; "
            "rm -f /tmp/libtpu_lockfile /dev/shm/tmp_r*/libtpu_lockfile; sleep 3",
        ],
        check=False,
    )


def build_v5p_env(tp_size: int, replica_idx: int, replicas_per_host: int, cache_tag: str) -> dict[str, str]:
    env = os.environ.copy()
    for k in (
        "JOB_COMPLETION_INDEX",
        "JOB_COMPLETION_MODE",
        "TPU_PROCESS_PORT",
        "TPU_PROCESS_ADDRESSES",
        "TPU_CHIPS_PER_PROCESS_BOUNDS",
        "TPU_PROCESS_BOUNDS",
        "TPU_WORKER_ID",
        "TPU_VISIBLE_CHIPS",
        "TPU_VISIBLE_DEVICES",
        "TPU_RUNTIME_METRICS_PORTS",
        "TPU_SKIP_MDS_QUERY",
        "VBAR_CONTROL_SERVICE_URL",
        "TPU_WORKER_HOSTNAMES",
        "TPU_TOPOLOGY_WRAP",
        "TPU_TOPOLOGY_ALT",
        "TPU_TOPOLOGY",
        "TOPOLOGY",
        "TPU_CHIPS_PER_HOST_BOUNDS",
        "CHIPS_PER_HOST_BOUNDS",
        "TPU_HOST_BOUNDS",
        "HOST_BOUNDS",
        "TPU_ACCELERATOR_TYPE",
        "TORCH_TPU_SLICEBUILDER_ADDRESSES",
        "TORCH_TPU_TOPOLOGY",
        "TORCH_TPU_DEVICES_PER_HOST",
        "SGLANG_TPU_PD_COLOCATED",
    ):
        env.pop(k, None)

    port_seed = 29600 + replica_idx * 32
    sb = ",".join(f"localhost:{port_seed + i}" for i in range(tp_size))
    metrics_ports = ",".join(str(8440 + replica_idx * 8 + k) for k in range(tp_size))
    tmp_dir = Path(f"/dev/shm/tmp_r{replica_idx}_tp{tp_size}")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    ind_dir = Path(f"/dev/shm/ind_{cache_tag}_r{replica_idx}")
    ind_dir.mkdir(parents=True, exist_ok=True)
    xla_cache = Path("/dev/shm/torch_tpu_cache/default")
    xla_cache.mkdir(parents=True, exist_ok=True)
    jax_cache = Path("/dev/shm/jax_compile_cache")
    jax_cache.mkdir(parents=True, exist_ok=True)

    env.update(
        {
            "LD_PRELOAD": "/workspace/metadata_hook.so",
            "MOCK_METADATA_PORT": "18080",
            "GCE_METADATA_HOST": "127.0.0.1:18080",
            "GCE_METADATA_IP": "127.0.0.1",
            "GOOGLE_CLOUD_PROJECT": "mock-tpu-project",
            "GCLOUD_PROJECT": "mock-tpu-project",
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(port_seed - 1),
            "TORCH_TPU_SLICEBUILDER_ADDRESSES": sb,
            "TPU_SKIP_MDS_QUERY": "1",
            "TPU_ACCELERATOR_TYPE": "v5p-8",
            "TPU_CHIPS_PER_HOST_BOUNDS": "2,2,1",
            "TPU_HOST_BOUNDS": "1,1,1",
            "TPU_TOPOLOGY": "2x2x1",
            "TPU_WORKER_ID": "0",
            "TPU_RUNTIME_METRICS_PORTS": metrics_ports,
            "SGLANG_EXTERNAL_MODEL_PACKAGE": "flint_sglang",
            "FLINT_TPU_ROUTER_MODE": "fused",
            "FLASHINFER_USE_CUDA_NORM": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "SGLANG_TPU_RELEASE_COMPILE_ARTIFACTS": "0",
            "SGLANG_TPU_RELEASE_CHECKPOINT_PAGE_CACHE": "0",
            "TORCH_TPU_CACHE_DIR": str(xla_cache),
            "TORCH_TPU_TIER2_COMPILATION_CACHE": "1",
            "TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT": str(xla_cache),
            "JAX_COMPILATION_CACHE_DIR": str(jax_cache),
            "SGLANG_BASE_TORCHINDUCTOR_CACHE_DIR": str(ind_dir),
            "TORCHINDUCTOR_CACHE_DIR": str(ind_dir),
            "TORCHINDUCTOR_AUTOGRAD_CACHE": "0",
            "OMP_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
            "TMPDIR": str(tmp_dir),
            "TPU_LOCKFILE_DIR": str(tmp_dir),
            "PYTHONUNBUFFERED": "1",
        }
    )
    if replicas_per_host > 1:
        env["SGLANG_TPU_PD_COLOCATED"] = "1"
    if tp_size == 4:
        env["TORCH_TPU_TOPOLOGY"] = "2,2,1"
    elif tp_size == 1:
        env["TORCH_TPU_TOPOLOGY"] = "1,1,1"
        env["TPU_CHIPS_PER_PROCESS_BOUNDS"] = "1,1,1"
        env["TPU_PROCESS_BOUNDS"] = "1,1,1"
        env["TPU_PROCESS_ADDRESSES"] = "local"
    if Path("/var/run/tpu-plugin/vbar.sock").is_socket():
        env["VBAR_CONTROL_SERVICE_URL"] = "unix:///var/run/tpu-plugin/vbar.sock"
    return env


def check_health(url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{url}/health", timeout=3.0) as r:
            return r.status == 200
    except Exception:
        return False


def launch_v5p_servers(
    idx: int,
    *,
    tag: str,
    model_path: str,
    tp_size: int,
    replicas_per_host: int,
    kv_cache_dtype: str = "fp8_e4m3",
    mem_fraction: float = 0.84,
    context_length: int = 32768,
    chunked_prefill: int = 8192,
    max_running_requests: int = 64,
    swa_ratio: float = 0.10,
    enable_compile: bool = True,
    extra_args: list[str] | None = None,
    timeout_s: int = 1500,
) -> tuple[bool, dict[str, Any]]:
    kill_sglang()
    Path("/tmp/v5p_mock_meta_state.json").write_text(
        json.dumps(
            {
                "chips_per_host_bounds": "2,2,1",
                "host_bounds": "1,1,1",
                "topology": "2x2x1",
                "accelerator_type": "v5p-8",
                "metrics_ports": "8431,8432,8433,8434",
            },
            indent=2,
        )
    )
    t0 = time.time()
    procs: list[subprocess.Popen] = []
    logs: list[Path] = []
    urls: list[str] = []
    cache_tag = f"tp{tp_size}_{'lora' if (extra_args and '--enable-lora' in extra_args) else 'base'}"
    for r in range(replicas_per_host):
        port = 30000 + r
        lp = Path(f"/workspace/logs/{tag}_r{r}.log")
        lp.parent.mkdir(parents=True, exist_ok=True)
        logs.append(lp)
        cmd = [
            sys.executable,
            "-m",
            "sglang.launch_server",
            "--device",
            "tpu",
            "--model-path",
            model_path,
            "--host",
            "0.0.0.0",
            "--port",
            str(port),
            "--tp-size",
            str(tp_size),
            "--base-gpu-id",
            str(r * tp_size),
            "--trust-remote-code",
            "--attention-backend",
            "pallas_tpu_rpa_v3",
            "--kv-cache-dtype",
            kv_cache_dtype,
            "--moe-runner-backend",
            "auto",
            "--grammar-backend",
            "flint",
            "--reasoning-parser",
            "flint",
            "--tool-call-parser",
            "flint",
            "--enable-fp32-lm-head",
            "--page-size",
            "64",
            "--mem-fraction-static",
            str(mem_fraction),
            "--max-running-requests",
            str(max_running_requests),
            "--chunked-prefill-size",
            str(chunked_prefill),
            "--context-length",
            str(context_length),
            "--swa-full-tokens-ratio",
            str(swa_ratio),
        ]
        if enable_compile:
            cmd.append("--enable-torch-compile")
        if extra_args:
            cmd.extend(extra_args)
        env = build_v5p_env(tp_size, r, replicas_per_host, cache_tag)
        lf = open(lp, "w")
        p = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
        procs.append(p)
        urls.append(f"http://127.0.0.1:{port}")
        if replicas_per_host > 1 and r < replicas_per_host - 1:
            time.sleep(6)

    ready = [False] * replicas_per_host
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        for r in range(replicas_per_host):
            if not ready[r]:
                if procs[r].poll() is not None:
                    tail = logs[r].read_text(errors="replace")[-2500:]
                    kill_sglang()
                    return False, {"error": f"replica {r} exited rc={procs[r].returncode}", "log_tail": tail}
                if check_health(urls[r]):
                    ready[r] = True
                    log(idx, f"[{tag}] Replica {r}/{replicas_per_host} READY in {time.time() - t0:.1f}s")
        if all(ready):
            break
        time.sleep(3)

    if not all(ready):
        tails = {f"r{r}": logs[r].read_text(errors="replace")[-1200:] for r in range(replicas_per_host)}
        kill_sglang()
        return False, {"error": "timeout", "log_tails": tails}

    entry_url = urls[0]
    if replicas_per_host > 1:
        subprocess.run(
            [
                sys.executable,
                "/workspace/gateway_flint_v7x.py",
                "--port",
                "8000",
                "--workers",
                ",".join(urls),
                "--daemon",
                "--log-file",
                f"/workspace/logs/gw_{tag}.log",
            ],
            check=True,
        )
        time.sleep(1.5)
        entry_url = "http://127.0.0.1:8000"

    return True, {
        "entry_url": entry_url,
        "backend_urls": urls,
        "boot_s": round(time.time() - t0, 2),
        "tp_size": tp_size,
        "replicas_per_host": replicas_per_host,
    }


async def _one_req(
    session: aiohttp.ClientSession,
    base_url: str,
    model_path: str,
    prompt: list[int] | str,
    osl: int,
    lora_name: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model_path,
        "prompt": prompt,
        "max_tokens": osl,
        "temperature": 0.0,
        "ignore_eos": True,
        "stream": True,
    }
    if lora_name is not None:
        payload["lora_path"] = lora_name
    t0 = time.perf_counter()
    first_t: float | None = None
    chunks = 0
    try:
        async with session.post(f"{base_url}/v1/completions", json=payload) as resp:
            if resp.status != 200:
                body = await resp.text()
                return {"ok": False, "error": f"HTTP {resp.status}: {body[:240]}"}
            async for raw in resp.content:
                line = raw.decode("utf-8", errors="ignore").strip()
                if not line.startswith("data:"):
                    continue
                d = line[5:].strip()
                if d == "[DONE]":
                    break
                try:
                    ev = json.loads(d)
                except Exception:
                    continue
                choices = ev.get("choices") or []
                if choices and choices[0].get("text"):
                    if first_t is None:
                        first_t = time.perf_counter()
                    chunks += 1
        t1 = time.perf_counter()
        ttft = (first_t - t0) if first_t else (t1 - t0)
        tpot = ((t1 - first_t) / max(osl - 1, 1)) if (first_t and osl > 1) else 0.0
        return {"ok": True, "ttft_s": ttft, "tpot_s": tpot, "e2e_s": t1 - t0, "output_tokens": osl, "chunks": chunks}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


async def run_point(
    base_url: str,
    model_path: str,
    isl: int,
    osl: int,
    concurrency: int,
    hit_rate: float = 0.0,
    lora_schedule: list[str | None] | None = None,
    use_code_prompt: bool = False,
) -> dict[str, Any]:
    rng = random.Random(2026 + isl + osl + concurrency)
    n_reqs = max(concurrency * 2, 8) if isl <= 8192 else max(concurrency, 2)
    shared_len = int(isl * hit_rate)
    shared_prefix = [rng.randint(100, 50000) for _ in range(shared_len)] if shared_len > 0 else []

    prompts: list[list[int] | str] = []
    if use_code_prompt:
        base_txt = CODE_PREFIX_TEXT[: max(isl * 3, 512)]
        for i in range(n_reqs):
            prompts.append(f"# Request {i}\n{base_txt}\n    def step_{i}(self, x: int) -> int:\n")
    else:
        for _ in range(n_reqs):
            tail = [rng.randint(100, 50000) for _ in range(isl - shared_len)]
            prompts.append(shared_prefix + tail)

    sched = lora_schedule or [None]
    timeout = aiohttp.ClientTimeout(total=1800)
    connector = aiohttp.TCPConnector(limit=max(concurrency * 2, 64))
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        # Warm prefix cache if hit_rate > 0
        if hit_rate > 0 and shared_prefix:
            await _one_req(session, base_url, model_path, prompts[0], 4, sched[0])
        sem = asyncio.Semaphore(concurrency)

        async def _worker(idx_r: int) -> dict[str, Any]:
            async with sem:
                ln = sched[idx_r % len(sched)]
                return await _one_req(session, base_url, model_path, prompts[idx_r], osl, ln)

        t_start = time.perf_counter()
        res = await asyncio.gather(*[_worker(i) for i in range(n_reqs)])
        wall = time.perf_counter() - t_start

    oks = [r for r in res if r.get("ok")]
    errs = [r for r in res if not r.get("ok")]
    total_out = sum(r["output_tokens"] for r in oks)
    ttfts = [r["ttft_s"] * 1000.0 for r in oks]
    tpots = [r["tpot_s"] * 1000.0 for r in oks if r["tpot_s"] > 0]
    p50_tpot = float(np.percentile(tpots, 50)) if tpots else 0.0
    return {
        "isl": isl,
        "osl": osl,
        "hit_rate": hit_rate,
        "concurrency": concurrency,
        "n_reqs": n_reqs,
        "completed": len(oks),
        "failed": len(errs),
        "wall_s": round(wall, 3),
        "host_output_tok_s": round(total_out / max(wall, 1e-6), 2),
        "slice_16chip_equiv_tok_s": round(4.0 * total_out / max(wall, 1e-6), 2),
        "ttft_ms_p50": round(float(np.percentile(ttfts, 50)), 2) if ttfts else 0.0,
        "ttft_ms_p99": round(float(np.percentile(ttfts, 99)), 2) if ttfts else 0.0,
        "tpot_ms_p50": round(p50_tpot, 3),
        "single_stream_tok_s": round(1000.0 / p50_tpot, 2) if p50_tpot > 0 else 0.0,
        "sample_error": errs[0]["error"] if errs else None,
    }


def http_post_json(url: str, payload: dict[str, Any], timeout: float = 90.0) -> tuple[int, Any]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read().decode("utf-8", errors="replace")
            try:
                return r.status, json.loads(body)
            except Exception:
                return r.status, body
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", errors="replace")[:400]
    except Exception as e:
        return -1, f"{type(e).__name__}: {e}"


def run_multilora_track(idx: int, precision: str, model_path: str, mem_frac: float) -> None:
    import patch_sglang_tpu_lora

    patch_sglang_tpu_lora.apply_sglang_tpu_lora_patches()
    staged = stage_adapters(Path("/dev/shm/flint_adapters"))
    names = list(staged.keys())[:4]
    lora_args = [
        "--enable-lora",
        "--lora-backend",
        "torch_native",
        "--max-loras-per-batch",
        "5",
        "--max-lora-rank",
        "16",
        "--lora-target-modules",
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "--lora-paths",
        *[f"{k}={staged[k]}" for k in names],
    ]
    tag = f"v5p_{precision}_multilora"
    out_file = RESULTS_DIR / f"pod{idx}_{tag}.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Launch TP=4 x DP=1 directly on 4-chip v5p host (single compiled process across all 4 chips)
    ok, info = launch_v5p_servers(
        idx,
        tag=f"{tag}_tp4dp1",
        model_path=model_path,
        tp_size=4,
        replicas_per_host=1,
        mem_fraction=mem_frac,
        context_length=69632,
        extra_args=lora_args,
        timeout_s=2400,
    )
    if not ok:
        log(idx, f"[{tag}] Retrying TP=4 with --disable-torch-compile...")
        ok, info = launch_v5p_servers(
            idx,
            tag=f"{tag}_tp4_eager",
            model_path=model_path,
            tp_size=4,
            replicas_per_host=1,
            mem_fraction=mem_frac,
            context_length=69632,
            enable_compile=False,
            extra_args=lora_args,
        )
    if not ok:
        out_file.write_text(json.dumps({"status": "failed", "tag": tag, "info": info}, indent=2) + "\n")
        return

    entry = info["entry_url"]
    schedules: list[tuple[str, list[str | None]]] = [
        ("0_loras_base_only", [None]),
        ("1_lora_bf16", [names[0]]),
        ("2_loras_bf16_plus_qlora_int8", names[:2]),
        ("4_loras_all_mixed", names[:4]),
        ("mixed_base_plus_3_loras", [None] + names[:3]),
    ]
    points: list[dict[str, Any]] = []
    for sname, sched in schedules:
        for isl, osl, concs, hit in [
            (1024, 512, [1, 8, 16, 32], 0.0),
            (4096, 512, [1, 16, 32], 0.95),
            (65536, 512, [1, 8], 0.95),
        ]:
            if isl >= 4096 and sname not in ("0_loras_base_only", "1_lora_bf16", "4_loras_all_mixed"):
                continue
            for c in concs:
                pt = asyncio.run(run_point(entry, model_path, isl, osl, c, hit_rate=hit, lora_schedule=sched))
                pt["schedule_name"] = sname
                points.append(pt)
                log(
                    idx,
                    f"[{tag}] {sname} {isl}:{osl} hit={int(hit*100)}% c={c} -> {pt['host_output_tok_s']} tok/s/host "
                    f"({pt['slice_16chip_equiv_tok_s']} 16-chip) | 1-stream={pt['single_stream_tok_s']} tok/s | errs={pt['failed']}",
                )
                out_file.write_text(
                    json.dumps(
                        {"status": "in_progress", "tag": tag, "precision": precision, "launch": info, "adapters": staged, "points": points},
                        indent=2,
                    )
                    + "\n"
                )

    # Quality probes + dynamic unload/reload on backendUrls
    probes: list[dict[str, Any]] = []
    for target in [None] + names[:3]:
        st_code, resp = http_post_json(
            f"{entry}/v1/chat/completions",
            {
                "model": model_path,
                "messages": [{"role": "user", "content": QUALITY_PROMPTS[0]}],
                "max_tokens": 160,
                "temperature": 0.0,
                **({"lora_path": target} if target else {}),
            },
        )
        txt = ""
        if isinstance(resp, dict) and resp.get("choices"):
            txt = resp["choices"][0].get("message", {}).get("content", "")
        probes.append({"adapter": target or "base", "http": st_code, "preview": txt[:280]})

    hotswap: dict[str, Any] = {}
    t_u0 = time.perf_counter()
    for bu in info["backend_urls"]:
        sc, _ = http_post_json(f"{bu}/unload_lora_adapter", {"lora_name": names[-1]})
        hotswap["unload_http"] = sc
    hotswap["unload_s"] = round(time.perf_counter() - t_u0, 3)
    t_l0 = time.perf_counter()
    for bu in info["backend_urls"]:
        sc, _ = http_post_json(f"{bu}/load_lora_adapter", {"lora_name": names[-1], "lora_path": staged[names[-1]]})
        hotswap["reload_http"] = sc
    hotswap["reload_s"] = round(time.perf_counter() - t_l0, 3)

    out_file.write_text(
        json.dumps(
            {
                "status": "completed",
                "tag": tag,
                "precision": precision,
                "hardware": "TPU v5p Single-Host (4 chips) + 4-Host v5p-16 Equivalent",
                "launch": info,
                "adapters": staged,
                "quality_probes": probes,
                "dynamic_hotswap": hotswap,
                "points": points,
            },
            indent=2,
        )
        + "\n"
    )
    kill_sglang()
    log(idx, f"[{tag}] COMPLETED -> {out_file}")


def run_specdec_track(idx: int) -> None:
    out_file = RESULTS_DIR / f"pod{idx}_v5p_tp_dp_and_specdec.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {"track": "v5p_tp_dp_precision_and_specdec", "configs": {}}

    # 1. Quick capability probe of NGRAM speculative decoding on sglang_tpu
    ok_ng, info_ng = launch_v5p_servers(
        idx,
        tag="v5p_fp8_tp4_ngram4_probe",
        model_path="/dev/shm/flint-fp8",
        tp_size=4,
        replicas_per_host=1,
        mem_fraction=0.84,
        context_length=32768,
        extra_args=["--speculative-algorithm", "NGRAM", "--speculative-num-draft-tokens", "4"],
        timeout_s=120,
    )
    results["configs"]["v5p_fp8_tp4_ngram4_probe"] = {"status": "ready" if ok_ng else "unsupported", "info": info_ng}
    out_file.write_text(json.dumps(results, indent=2) + "\n")
    kill_sglang()

    # 2. High-value v5p single-host TP/DP & BF16 vs Block-FP8 scaling sweep (4 chips/host)
    configs = [
        ("v5p_fp8_tp1_dp4", "/dev/shm/flint-fp8", "fp8_block128", 1, 4, 0.80),
        ("v5p_fp8_tp2_dp2", "/dev/shm/flint-fp8", "fp8_block128", 2, 2, 0.84),
        ("v5p_bf16_tp2_dp2", "/dev/shm/flint", "bfloat16", 2, 2, 0.82),
        ("v5p_bf16_tp4_dp1", "/dev/shm/flint", "bfloat16", 4, 1, 0.86),
    ]
    for ctag, mpath, prec, tp_s, reps, mfrac in configs:
        ok, info = launch_v5p_servers(
            idx,
            tag=ctag,
            model_path=mpath,
            tp_size=tp_s,
            replicas_per_host=reps,
            mem_fraction=mfrac,
            context_length=69632,
        )
        if not ok:
            results["configs"][ctag] = {"status": "unsupported_or_failed", "precision": prec, "info": info}
            out_file.write_text(json.dumps(results, indent=2) + "\n")
            continue
        pts = []
        for isl, osl, hit, concs in [
            (1024, 512, 0.0, [1, 8, 16, 32, 64]),
            (4096, 512, 0.95, [1, 16, 32, 64]),
            (65536, 512, 0.95, [1, 8, 16]),
        ]:
            for c in concs:
                pt = asyncio.run(run_point(info["entry_url"], mpath, isl, osl, c, hit_rate=hit))
                pts.append(pt)
                log(
                    idx,
                    f"[{ctag}] {isl}:{osl} hit={int(hit*100)}% c={c} -> {pt['host_output_tok_s']} tok/s/host "
                    f"({pt['slice_16chip_equiv_tok_s']} 16-chip) | 1-stream={pt['single_stream_tok_s']} tok/s | errs={pt['failed']}",
                )
            results["configs"][ctag] = {"status": "in_progress", "precision": prec, "launch": info, "points": pts}
            out_file.write_text(json.dumps(results, indent=2) + "\n")
        results["configs"][ctag]["status"] = "completed"
        out_file.write_text(json.dumps(results, indent=2) + "\n")
        kill_sglang()
    log(idx, f"TP/DP + SpecDec track COMPLETED -> {out_file}")


def run_radix_longctx_track(idx: int) -> None:
    out_file = RESULTS_DIR / f"pod{idx}_v5p_radix_swa_longctx.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    configs = [
        ("v5p_fp8_tp4_radix_on_swa010", "/dev/shm/flint-fp8", "fp8_e4m3", 0.10, []),
        ("v5p_fp8_tp4_radix_off_swa010", "/dev/shm/flint-fp8", "fp8_e4m3", 0.10, ["--disable-radix-cache"]),
        ("v5p_fp8_tp4_swa005_longctx", "/dev/shm/flint-fp8", "fp8_e4m3", 0.05, []),
        ("v5p_fp8_tp4_bf16kv_longctx", "/dev/shm/flint-fp8", "bfloat16", 0.10, []),
    ]
    results: dict[str, Any] = {"track": "v5p_radix_swa_longctx_128k", "configs": {}}
    for ctag, mpath, kv_dt, swa_r, extra in configs:
        ok, info = launch_v5p_servers(
            idx,
            tag=ctag,
            model_path=mpath,
            tp_size=4,
            replicas_per_host=1,
            kv_cache_dtype=kv_dt,
            mem_fraction=0.86,
            context_length=135168,
            swa_ratio=swa_r,
            extra_args=extra,
        )
        if not ok:
            results["configs"][ctag] = {"status": "failed", "info": info}
            out_file.write_text(json.dumps(results, indent=2) + "\n")
            continue
        pts = []
        grid = [
            (4096, 512, 0.0, [1, 16, 32]),
            (4096, 512, 0.95, [1, 16, 32]),
            (65536, 512, 0.0, [1, 8]),
            (65536, 512, 0.95, [1, 8, 16]),
            (131072, 512, 0.0, [1, 4]),
            (131072, 512, 0.95, [1, 8, 16]),
        ]
        for isl, osl, hit, concs in grid:
            for c in concs:
                pt = asyncio.run(run_point(info["entry_url"], mpath, isl, osl, c, hit_rate=hit))
                pts.append(pt)
                log(
                    idx,
                    f"[{ctag}] {isl}:{osl} hit={int(hit*100)}% c={c} -> {pt['host_output_tok_s']} tok/s | TTFT={pt['ttft_ms_p50']}ms | 1-stream={pt['single_stream_tok_s']} tok/s | errs={pt['failed']}",
                )
            results["configs"][ctag] = {"status": "in_progress", "kv_cache_dtype": kv_dt, "swa_ratio": swa_r, "launch": info, "points": pts}
            out_file.write_text(json.dumps(results, indent=2) + "\n")
        results["configs"][ctag]["status"] = "completed"
        out_file.write_text(json.dumps(results, indent=2) + "\n")
        kill_sglang()
    log(idx, f"Radix/SWA/128K track COMPLETED -> {out_file}")


def main() -> None:
    idx = int(os.environ.get("JOB_COMPLETION_INDEX", "0"))
    log(idx, f"Booting Wave B worker on host={os.uname().nodename} idx={idx}")
    sys.path.insert(0, "/workspace")

    # Stage checkpoints needed by this pod
    if idx in (0, 2, 3):
        stage_tree("/models/models/RecursiveSI/flint-fp8", "/dev/shm/flint-fp8", idx)
    if idx in (1, 2):
        stage_tree("/models/models/RecursiveSI/flint", "/dev/shm/flint", idx)

    if idx == 0:
        run_multilora_track(0, "fp8_block128", "/dev/shm/flint-fp8", 0.82)
    elif idx == 1:
        run_multilora_track(1, "bfloat16", "/dev/shm/flint", 0.80)
    elif idx == 2:
        run_specdec_track(2)
    elif idx == 3:
        run_radix_longctx_track(3)


if __name__ == "__main__":
    main()
