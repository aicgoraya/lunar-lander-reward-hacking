#!/bin/bash
# How big does the hover bonus have to be before the agent stops landing?
# Runs 2 trainings at a time, then evaluates each on the plain game.
cd "$(dirname "$0")"
mkdir -p logs models results videos
jobs_list=()
for b in 0.02 0.05 0.1 0.2 0.3 0.5; do
  for s in 0 1; do jobs_list+=("$b $s"); done
done
run() {
  b=$1; s=$2; tag="hover_${b}_s${s}"
  python3 train.py "hover:$b" 1000000 "$s" > "logs/hover_${b}_s${s}.txt" 2>&1
  python3 evaluate.py "$tag" 30 > "logs/eval_hover_${b}_s${s}.txt" 2>&1
}
i=0
for j in "${jobs_list[@]}"; do
  run $j &
  i=$((i+1))
  if (( i % 2 == 0 )); then wait; fi
done
wait
echo SWEEP_DONE
