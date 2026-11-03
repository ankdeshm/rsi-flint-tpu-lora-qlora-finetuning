#!/bin/bash
# Per-host launcher for Flint 40B MoE LoRA / QLoRA fine-tuning on TPU v5p-16 (4 hosts x 4 chips).
# usage: launch_flint_ft.sh <config> <run_name> [extra torchtitan CLI overrides...]
set -uo pipefail
CONFIG=$1
RUN=$2
shift 2
NODE_RANK=${JOB_COMPLETION_INDEX:-0}
NNODES=${LORA_FT_NNODES:-4}
NPROC=${LORA_FT_NPROC:-4}
MASTER=${LORA_FT_MASTER:-flint-lora-ft-v5p16-0.flint-lora-ft-svc.default.svc.cluster.local}
PORT=${LORA_FT_PORT:-29515}
CODE=${LORA_FT_CODE:-/dev/shm/code}
LOGDIR=${LORA_FT_LOGDIR:-/dev/shm/logs}
export PYTHONPATH=${CODE}:/torchtitan-native
export LORA_FT_RUN_NAME=${RUN}
export LORA_FT_TS=${LORA_FT_TS:-$(date -u +%Y%m%dT%H%M%SZ)}
export PJRT_DEVICE=TPU
mkdir -p "${LOGDIR}" /models/flint-lora-ft/logs
LOG=${LOGDIR}/${RUN}.node${NODE_RANK}.log
cd /torchtitan-native

HBM_PID=""
if [ "${NODE_RANK}" = "0" ] && [ -x "${CODE}/hbm_sampler.sh" ]; then
  "${CODE}/hbm_sampler.sh" "${LOGDIR}/${RUN}.hbm.log" 7200 &
  HBM_PID=$!
fi

echo "[launch_flint_ft] $(date -u +%FT%TZ) host=$(hostname) node_rank=${NODE_RANK} config=${CONFIG} run=${RUN} master=${MASTER}:${PORT} extra=$*" > "${LOG}"
torchrun --nnodes="${NNODES}" --nproc_per_node="${NPROC}" --rdzv_backend=static \
  --rdzv_endpoint="${MASTER}:${PORT}" --node_rank="${NODE_RANK}" \
  -m torchtitan.experiments.tpu.train --module=flint_lora_tpu_recipes --config="${CONFIG}" "$@" >> "${LOG}" 2>&1
rc=$?
echo "[launch_flint_ft] rc=${rc} $(date -u +%FT%TZ)" >> "${LOG}"

if [ -n "${HBM_PID}" ]; then
  kill "${HBM_PID}" 2>/dev/null || true
  cp -f "${LOGDIR}/${RUN}.hbm.log" "/models/flint-lora-ft/logs/${RUN}.hbm.log" || true
fi
cp -f "${LOG}" "/models/flint-lora-ft/logs/${RUN}.node${NODE_RANK}.log" || true
exit ${rc}
