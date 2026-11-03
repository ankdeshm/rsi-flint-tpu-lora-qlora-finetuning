#!/usr/bin/env python3
"""Parses TorchTitan + TorchTPU 0.2.0 Flint 40B MoE LoRA & QLoRA fine-tuning logs and tpu-info HBM telemetry."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path
from typing import Any


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
STEP_RE = re.compile(
    r"step:\s*(?P<step>\d+)\s+loss:\s*(?P<loss>[0-9.]+)\s+grad_norm:\s*(?P<grad_norm>[0-9.]+)\s+"
    r"memory:\s*(?P<mem_gib>[0-9.]+)GiB\((?P<mem_pct>[0-9.]+)%\)\s+"
    r"tps:\s*(?P<tps>[0-9,]+)\s+tflops:\s*(?P<tflops>[0-9.]+)\s+mfu:\s*(?P<mfu>[0-9.]+)%\s+"
    r"time_end_to_end:\s*(?P<step_s>[0-9.]+)s"
)


def parse_node0_log(log_path: Path, steady_min_step: int = 15) -> dict[str, Any]:
    by_step: dict[int, dict[str, float]] = {}
    rc: int | None = None
    for raw in log_path.read_text(errors="replace").splitlines():
        line = ANSI_RE.sub("", raw)
        m = STEP_RE.search(line)
        if m:
            st = int(m.group("step"))
            by_step[st] = {
                "step": st,
                "loss": float(m.group("loss")),
                "grad_norm": float(m.group("grad_norm")),
                "pytorch_mem_gib": float(m.group("mem_gib")),
                "pytorch_mem_pct": float(m.group("mem_pct")),
                "tok_s_per_chip": float(m.group("tps").replace(",", "")),
                "tflops_per_chip": float(m.group("tflops")),
                "active_mfu_pct": float(m.group("mfu")),
                "step_time_s": float(m.group("step_s")),
            }
        if "[launch_flint_ft] rc=" in line:
            try:
                rc = int(line.split("rc=")[1].split()[0])
            except Exception:
                pass

    steps_sorted = [by_step[k] for k in sorted(by_step)]
    steady = [x for x in steps_sorted if x["step"] >= steady_min_step]
    if not steady and len(steps_sorted) > 2:
        steady = steps_sorted[2:]
    elif not steady:
        steady = steps_sorted

    step_times = [x["step_time_s"] for x in steady]
    chip_tps = [x["tok_s_per_chip"] for x in steady]
    tflops = [x["tflops_per_chip"] for x in steady]
    mfus = [x["active_mfu_pct"] for x in steady]
    mems = [x["pytorch_mem_gib"] for x in steps_sorted]

    checkpoints = {}
    for target_s in (1, 5, 10, 25, 50, 75, 100, 120, 150):
        if target_s in by_step:
            checkpoints[f"step_{target_s}"] = round(by_step[target_s]["loss"], 5)

    mean_chip_tps = statistics.mean(chip_tps) if chip_tps else 0.0
    mean_tflops = statistics.mean(tflops) if tflops else 0.0
    mean_mfu = statistics.mean(mfus) if mfus else 0.0
    return {
        "log_file": log_path.name,
        "return_code": rc,
        "total_steps_logged": len(steps_sorted),
        "compile_step1_s": round(by_step[1]["step_time_s"], 3) if 1 in by_step else None,
        "compile_step2_s": round(by_step[2]["step_time_s"], 3) if 2 in by_step else None,
        "steady_min_step": steady_min_step,
        "steady_num_steps": len(steady),
        "steady_step_time_mean_s": round(statistics.mean(step_times), 4) if step_times else 0.0,
        "steady_step_time_p50_s": round(statistics.median(step_times), 4) if step_times else 0.0,
        "steady_step_time_min_s": round(min(step_times), 4) if step_times else 0.0,
        "steady_tok_s_per_chip": round(mean_chip_tps, 1),
        "steady_slice_tok_s_16chips": round(mean_chip_tps * 16.0, 1),
        "steady_tflops_per_chip": round(mean_tflops, 2),
        "steady_active_mfu_pct": round(mean_mfu, 2),
        # With Full Activation Checkpointing (4/3x FLOPs vs 6*N_active), HFU = 4/3 * MFU
        "steady_active_hfu_pct": round(mean_mfu * (4.0 / 3.0), 2),
        "pytorch_peak_mem_gib": round(max(mems), 3) if mems else 0.0,
        "loss_initial": round(steps_sorted[0]["loss"], 5) if steps_sorted else None,
        "loss_final": round(steps_sorted[-1]["loss"], 5) if steps_sorted else None,
        "loss_last10_mean": round(statistics.mean(x["loss"] for x in steps_sorted[-10:]), 5) if len(steps_sorted) >= 10 else None,
        "loss_curve_checkpoints": checkpoints,
    }


def parse_hbm_log(hbm_path: Path) -> dict[str, Any] | None:
    if not hbm_path.exists() or hbm_path.stat().st_size == 0:
        return None
    samples = []
    for ln in hbm_path.read_text(errors="replace").splitlines():
        parts = ln.strip().split()
        if len(parts) >= 5 and all(re.match(r"^[0-9.]+$", p) for p in parts[1:5]):
            vals = [float(p) for p in parts[1:5]]
            samples.append({"ts": parts[0], "chips_gib": vals})
    if not samples:
        return None
    flat = sorted(v for s in samples for v in s["chips_gib"])
    # Exclude initial load/shutdown zeros (< 2.0 GiB) when computing active runtime resting/peak
    active = [v for v in flat if v >= 2.0] or flat
    p10 = active[max(0, int(0.10 * len(active)) - 1)]
    p50 = statistics.median(active)
    p95 = active[min(len(active) - 1, int(0.95 * len(active)))]
    return {
        "hbm_file": hbm_path.name,
        "tool": "tpu-info (libtpu runtime HBM usage, 4 chips on host 0, 2s interval)",
        "num_timestamps": len(samples),
        "resting_p10_gib": round(p10, 2),
        "median_gib": round(p50, 2),
        "p95_gib": round(p95, 2),
        "max_gib": round(max(active), 2),
        "sample_window": samples[len(samples) // 3 : len(samples) // 3 + 15],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs-dir", required=True)
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    logs_dir = Path(args.logs_dir)
    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_names = [
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
    ]
    summary_runs: dict[str, Any] = {}
    for rname in run_names:
        log_p = logs_dir / f"{rname}.node0.log"
        if not log_p.exists():
            continue
        min_st = 10 if rname.endswith(("_16k", "_32k", "_64k")) else 15
        parsed = parse_node0_log(log_p, steady_min_step=min_st)
        # Match latest run_stats.json for this exact run_name (timestamp starts with _202)
        cand_dirs = sorted(runs_dir.glob(f"{rname}_202*"))
        if cand_dirs:
            rs_path = cand_dirs[-1] / "run_stats.json"
            if rs_path.exists():
                rs = json.loads(rs_path.read_text())
                parsed["run_stats"] = rs
                (out_dir / f"{rname}_run_stats.json").write_text(json.dumps(rs, indent=2) + "\n")
        hbm = parse_hbm_log(logs_dir / f"{rname}.hbm.log")
        if hbm is not None:
            parsed["tpu_info_hbm"] = {k: v for k, v in hbm.items() if k != "sample_window"}
            (out_dir / f"{rname}_hbm_telemetry.json").write_text(json.dumps(hbm, indent=2) + "\n")
        summary_runs[rname] = parsed

    master = {
        "title": "RecursiveSI/flint (39.81B total / 2.52B active MoE) LoRA & QLoRA Fine-Tuning on TPU v5p-16",
        "hardware": {
            "slice": "TPU v5p-16 (2x2x4, 4 hosts x 4 chips = 16 chips)",
            "hbm_per_chip_gib": 95.0,
            "bf16_peak_tflops_per_chip": 459.0,
        },
        "dataset": {
            "name": "yahma/alpaca-cleaned",
            "seq_len": 4096,
            "local_batch_size": 2,
            "global_tokens_per_step": 131072,
        },
        "runs": summary_runs,
    }
    (out_dir / "summary.json").write_text(json.dumps(master, indent=2) + "\n")
    print(f"Wrote summary.json with {len(summary_runs)} runs to {out_dir}")


if __name__ == "__main__":
    main()
