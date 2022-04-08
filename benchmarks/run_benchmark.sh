#!/bin/bash

function print_usage {
  echo -e "./run_benchmark.sh <name> (name must be the relative dir you want all the benchmarks to end up in)"
  echo -e "EXAMPLE ./run_benchmark.sh static_budget/total_limit/limit_20M_local_mongod"
}

if [[ $# -ne 1 ]]; then
  print_usage
  exit 1
fi

BENCHMARK_NAME="$1"
mkdir -p "$BENCHMARK_NAME"

echo "Starting omni for $BENCHMARK_NAME benchmark..."
./start_omni.sh "$BENCHMARK_NAME"
./start_free.sh

sleep 5

python3.8 make_request.py "$BENCHMARK_NAME" > "$BENCHMARK_NAME/response.json"

sleep 5

./stop_omni.sh
./stop_free.sh "$BENCHMARK_NAME"

if [[ -f "/s/parsons/b/others/sustain/local-disk/a/tmp/intermediate_response.json" ]]; then
  mv "/s/parsons/b/others/sustain/local-disk/a/tmp/intermediate_response.json" "$BENCHMARK_NAME/intermediate_response.json"
  mv "/s/parsons/b/others/sustain/local-disk/a/tmp/optimal_allocations.json" "$BENCHMARK_NAME/optimal_allocations.json"
  mv "/s/parsons/b/others/sustain/local-disk/a/tmp/numpy_array.json" "$BENCHMARK_NAME/numpy_array.json"
fi
