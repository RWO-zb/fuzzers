#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Please enter the RL key: $0 {bw|ll|tt}"
    exit 1
fi

if [[ "$1" != "bw" && "$1" != "ll" && "$1" != "tt" ]]; then
    echo "Invalid RL key: $1. Expected one of {bw|ll|tt}"
    exit 1
fi

script_name="python test_mdpfuzz_rl.py"

scripts=()

for ((i = 0; i < 110; i++)); do
    scripts+=("python test_mdpfuzz_rl.py $1 $i ../../data_rq3/$1")
done

max_processes="${#scripts[@]}"
echo "launching $max_processes..."
parallel --jobs $max_processes ::: "${scripts[@]}"