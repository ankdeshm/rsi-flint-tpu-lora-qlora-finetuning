#!/usr/bin/env python3
"""Multi-host controller for `job/flint-lora-ft-v5p16` (4 hosts x 4 v5p chips = 16 chips)."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import sys
import time
from pathlib import Path

CTX = "gke_your-gcp-project-id_us-central1_tpu-v5p-cluster"
CODE_SRC = Path("/Users/tpu-engineer/.gemini/jetski/brain/e9d0bbdf-50f1-475d-8d70-1183a4a99adf/scratch/flint_ft")
HBM_SAMPLER = Path("/Users/tpu-engineer/.gemini/jetski/brain/e9d0bbdf-50f1-475d-8d70-1183a4a99adf/scratch/harvest/lora_ft/code/hbm_sampler.sh")


_POD0_IP: str = ""


def get_pods() -> list[tuple[int, str]]:
    global _POD0_IP
    out = subprocess.check_output(
        ["kubectl", f"--context={CTX}", "get", "pods", "-l", "job-name=flint-lora-ft-v5p16", "-o", "json"],
        text=True,
    )
    items = json.loads(out)["items"]
    pods: list[tuple[int, str]] = []
    for it in items:
        if it["status"].get("phase") == "Running":
            idx = int(it["metadata"]["labels"].get("batch.kubernetes.io/job-completion-index", "0"))
            pods.append((idx, it["metadata"]["name"]))
            if idx == 0:
                _POD0_IP = it["status"].get("podIP", "")
    pods.sort()
    assert len(pods) == 4, f"Expected 4 Running pods, got: {pods}"
    return pods


def sync_code(pods: list[tuple[int, str]]) -> None:
    files = [CODE_SRC / "flint_lora_tpu_recipes.py", CODE_SRC / "launch_flint_ft.sh", HBM_SAMPLER]
    for extra_f in ("train_flint_eagle_draft.py", "launch_eagle_draft.sh"):
        if (CODE_SRC / extra_f).exists():
            files.append(CODE_SRC / extra_f)

    def sync_one(pod: str) -> None:
        subprocess.run(
            ["kubectl", f"--context={CTX}", "exec", pod, "-c", "trainer", "--", "mkdir", "-p", "/dev/shm/code", "/dev/shm/logs"],
            check=True,
        )
        for f in files:
            subprocess.run(
                ["kubectl", f"--context={CTX}", "cp", str(f), f"{pod}:/dev/shm/code/{f.name}", "-c", "trainer"],
                check=True,
            )
        subprocess.run(
            ["kubectl", f"--context={CTX}", "exec", pod, "-c", "trainer", "--", "bash", "-c", "chmod +x /dev/shm/code/*.sh"],
            check=True,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        list(ex.map(lambda p: sync_one(p[1]), pods))
    print(f"[ft_ctl] Synced code to {[p[1] for p in pods]}", flush=True)


def kill_all(pods: list[tuple[int, str]]) -> None:
    def k_one(pod: str) -> None:
        subprocess.run(
            [
                "kubectl",
                f"--context={CTX}",
                "exec",
                pod,
                "-c",
                "trainer",
                "--",
                "bash",
                "-c",
                "pkill -9 -f '[t]orchtitan.experiments.tpu.train' 2>/dev/null || true; pkill -9 -f '[t]rain_flint_eagle_draft' 2>/dev/null || true; pkill -9 -f '[t]orchrun' 2>/dev/null || true; pkill -9 -f '[h]bm_sampler.sh' 2>/dev/null || true",
            ],
            check=False,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        list(ex.map(lambda p: k_one(p[1]), pods))
    time.sleep(2.0)


def run_job(pods: list[tuple[int, str]], config: str, run_name: str, extra: str = "", env_kv: str = "", port: int = 29515) -> int:
    kill_all(pods)
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    m_env = f"LORA_FT_MASTER={_POD0_IP} " if _POD0_IP else ""

    def launch_pod(item: tuple[int, str]) -> tuple[int, int]:
        idx, pod = item
        cmd_str = (
            f"export {m_env}LORA_FT_TS={ts} LORA_FT_PORT={port} {env_kv}; "
            f"/dev/shm/code/launch_flint_ft.sh {config} {run_name} {extra}"
        )
        proc = subprocess.run(
            ["kubectl", f"--context={CTX}", "exec", pod, "-c", "trainer", "--", "bash", "-c", cmd_str],
            text=True,
            capture_output=True,
        )
        return idx, proc.returncode

    print(f"[ft_ctl] Launching config={config} run={run_name} ts={ts} master={_POD0_IP}:{port} extra='{extra}' env='{env_kv}'...", flush=True)
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        rcs = sorted(ex.map(launch_pod, pods))
    dt = time.time() - t0
    print(f"[ft_ctl] Finished run={run_name} in {dt:.1f}s rcs={rcs}", flush=True)
    tail = subprocess.check_output(
        ["kubectl", f"--context={CTX}", "exec", pods[0][1], "-c", "trainer", "--", "tail", "-n", "30", f"/dev/shm/logs/{run_name}.node0.log"],
        text=True,
    )
    print(tail, flush=True)
    return max(rc for _, rc in rcs)


def run_eagle_draft(pods: list[tuple[int, str]], extra: str = "", env_kv: str = "", port: int = 29560) -> int:
    kill_all(pods)
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    m_env = f"LORA_FT_MASTER={_POD0_IP} " if _POD0_IP else ""

    def launch_pod(item: tuple[int, str]) -> tuple[int, int]:
        idx, pod = item
        cmd_str = (
            f"export {m_env}LORA_FT_TS={ts} LORA_FT_PORT={port} {env_kv}; "
            f"/dev/shm/code/launch_eagle_draft.sh {extra}"
        )
        proc = subprocess.run(
            ["kubectl", f"--context={CTX}", "exec", pod, "-c", "trainer", "--", "bash", "-c", cmd_str],
            text=True,
            capture_output=True,
        )
        return idx, proc.returncode

    print(f"[ft_ctl] Launching EAGLE3 draft training ts={ts} extra='{extra}'...", flush=True)
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        rcs = sorted(ex.map(launch_pod, pods))
    dt = time.time() - t0
    print(f"[ft_ctl] Finished EAGLE3 draft training in {dt:.1f}s rcs={rcs}", flush=True)
    tail = subprocess.check_output(
        ["kubectl", f"--context={CTX}", "exec", pods[0][1], "-c", "trainer", "--", "tail", "-n", "40", "/dev/shm/logs/flint_eagle_draft.node0.log"],
        text=True,
    )
    print(tail, flush=True)
    return max(rc for _, rc in rcs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["sync", "kill", "run", "eagle_draft", "suite", "suite40b", "suite_longctx", "suite_128k", "phase3_all"])
    ap.add_argument("--config", default="")
    ap.add_argument("--run-name", default="")
    ap.add_argument("--extra", default="")
    ap.add_argument("--env", default="")
    ap.add_argument("--port", type=int, default=29515)
    args = ap.parse_args()

    pods = get_pods()
    if args.action == "sync":
        sync_code(pods)
    elif args.action == "kill":
        kill_all(pods)
    elif args.action == "run":
        sync_code(pods)
        rc = run_job(pods, args.config, args.run_name or args.config, args.extra, args.env, args.port)
        sys.exit(rc)
    elif args.action == "eagle_draft":
        sync_code(pods)
        rc = run_eagle_draft(pods, args.extra, args.env, args.port)
        sys.exit(rc)
    elif args.action in ("suite", "suite40b", "suite_longctx", "suite_128k", "phase3_all"):
        sync_code(pods)
        if args.action == "phase3_all":
            chk_e = subprocess.run(
                ["kubectl", f"--context={CTX}", "exec", pods[0][1], "-c", "trainer", "--", "bash", "-c", "ls /models/adapters/flint-40b-eagle-draft/training_stats.json >/dev/null 2>&1"],
                check=False,
            )
            if chk_e.returncode != 0:
                rc = run_eagle_draft(pods, args.extra, args.env, 29560)
                if rc != 0:
                    print(f"[ft_ctl] WARNING: eagle_draft exited with rc={rc}", flush=True)
                pods = get_pods()
            else:
                print("[ft_ctl] flint-40b-eagle-draft already exported on GCS, proceeding to suite_128k.", flush=True)

        if args.action == "suite":
            for i, dbg in enumerate(["flint_debug_lora_bf16", "flint_debug_qlora_int8", "flint_debug_qlora_nf4"]):
                rc = run_job(pods, dbg, dbg, port=29520 + i)
                if rc != 0:
                    print(f"[ft_ctl] Smoke test {dbg} failed with rc={rc}!", flush=True)
                    sys.exit(rc)
        if args.action in ("suite_128k", "phase3_all"):
            suite_runs = [
                ("flint_40b_lora_bf16_128k", "flint_40b_lora_bf16_128k", "--training.steps=20", "LORA_FT_ADAPTER_ALIAS=flint-40b-lora-128k", 29550),
                ("flint_40b_qlora_int8_128k", "flint_40b_qlora_int8_128k", "--training.steps=20", "LORA_FT_ADAPTER_ALIAS=flint-40b-qlora-int8-128k", 29551),
                ("flint_40b_qlora_nf4_128k", "flint_40b_qlora_nf4_128k", "--training.steps=20", "LORA_FT_ADAPTER_ALIAS=flint-40b-qlora-nf4-128k", 29552),
            ]
        elif args.action == "suite_longctx":
            suite_runs = [
                ("flint_40b_lora_bf16_16k", "flint_40b_lora_bf16_16k", "--training.steps=40", "LORA_FT_ADAPTER_ALIAS=flint-40b-lora-16k", 29540),
                ("flint_40b_qlora_int8_16k", "flint_40b_qlora_int8_16k", "--training.steps=40", "LORA_FT_ADAPTER_ALIAS=flint-40b-qlora-int8-16k", 29541),
                ("flint_40b_lora_bf16_32k", "flint_40b_lora_bf16_32k", "--training.steps=40", "LORA_FT_ADAPTER_ALIAS=flint-40b-lora-32k", 29542),
                ("flint_40b_qlora_int8_32k", "flint_40b_qlora_int8_32k", "--training.steps=40", "LORA_FT_ADAPTER_ALIAS=flint-40b-qlora-int8-32k", 29543),
                ("flint_40b_lora_bf16_64k", "flint_40b_lora_bf16_64k", "--training.steps=30", "LORA_FT_ADAPTER_ALIAS=flint-40b-lora-64k", 29544),
                ("flint_40b_qlora_int8_64k", "flint_40b_qlora_int8_64k", "--training.steps=30", "LORA_FT_ADAPTER_ALIAS=flint-40b-qlora-int8-64k", 29545),
            ]
        else:
            suite_runs = [
                ("flint_40b_lora_bf16", "flint_40b_lora_bf16", "--training.steps=150", "LORA_FT_ADAPTER_ALIAS=flint-40b-lora-r16", 29530),
                ("flint_40b_qlora_int8", "flint_40b_qlora_int8", "--training.steps=150", "LORA_FT_ADAPTER_ALIAS=flint-40b-qlora-int8-r16", 29531),
                ("flint_40b_qlora_nf4", "flint_40b_qlora_nf4", "--training.steps=150", "LORA_FT_ADAPTER_ALIAS=flint-40b-qlora-nf4-r16", 29532),
                ("flint_40b_lora_bf16_r64", "flint_40b_lora_bf16_r64", "--training.steps=120", "LORA_FT_ADAPTER_ALIAS=flint-40b-lora-r64", 29533),
                ("flint_40b_lora_bf16_attn", "flint_40b_lora_bf16_attn", "--training.steps=120", "LORA_FT_ADAPTER_ALIAS=flint-40b-lora-attn-r16", 29534),
            ]
        for cfg_name, rname, extra, env_kv, port in suite_runs:
            chk = subprocess.run(
                ["kubectl", f"--context={CTX}", "exec", pods[0][1], "-c", "trainer", "--", "bash", "-c", f"ls /models/flint-lora-ft/runs/{rname}_202*/run_stats.json >/dev/null 2>&1"],
                check=False,
            )
            if chk.returncode == 0:
                print(f"[ft_ctl] {rname} already completed on GCS, skipping.", flush=True)
                continue
            rc = run_job(pods, cfg_name, rname, extra=extra, env_kv=env_kv, port=port)
            if rc != 0:
                print(f"[ft_ctl] WARNING: {rname} exited with rc={rc}", flush=True)
            pods = get_pods()


if __name__ == "__main__":
    main()
