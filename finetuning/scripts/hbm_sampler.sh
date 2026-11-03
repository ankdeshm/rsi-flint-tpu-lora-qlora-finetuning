#!/bin/bash
# Samples libtpu runtime HBM usage per chip (via `tpu-info --metric hbm_usage`) every ~2 s while a run is
# active on this host; writes "<utc> chip0 chip1 chip2 chip3" (GiB used) lines to <out>.
# usage: hbm_sampler.sh <out_file> [max_seconds]
OUT=$1
MAX=${2:-7200}
END=$(( $(date +%s) + MAX ))
: > "${OUT}"
while [ "$(date +%s)" -lt "${END}" ]; do
  vals=$(timeout 20 tpu-info --metric hbm_usage 2>/dev/null | grep -E "^\| [0-9]+ +\|" | awk -F'|' '{print $3}' | awk '{print $1}' | tr '\n' ' ')
  [ -n "${vals}" ] && echo "$(date -u +%FT%TZ) ${vals}" >> "${OUT}"
  sleep 2
done
