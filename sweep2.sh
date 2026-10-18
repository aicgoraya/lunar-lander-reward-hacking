#!/bin/bash
cd "$(dirname "$0")"
run() { b=$1; s=$2; python3 train.py "hover:$b" 1000000 "$s" > "logs/hover_${b}_s${s}.txt" 2>&1; python3 evaluate.py "hover_${b}_s${s}" 30 > "logs/eval_hover_${b}_s${s}.txt" 2>&1; }
run 0.7 0 & run 0.7 1 & wait
run 0.85 0 & run 0.85 1 & wait
echo SWEEP2_DONE
