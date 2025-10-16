#!/bin/bash

script_name="python test_rl.py"

scripts=()

for ((i = 0; i < 15; i++)); do
    scripts+=("python test_rl.py ../../data_rq2/ $i bw")
    scripts+=("python test_rl.py ../../data_rq2/ $i ll")
    scripts+=("python test_rl.py ../../data_rq2/ $i tt")
done


max_processes="${#scripts[@]}"
echo "launching $max_processes..."
parallel --jobs $max_processes ::: "${scripts[@]}"