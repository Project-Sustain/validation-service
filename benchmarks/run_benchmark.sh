#!/bin/bash

function print_usage {
  echo -e "./run_benchmark.sh <name>"
  echo -e "EXAMPLE ./run_benchmark.sh sh13rs_gisjoins_vs"
}

if [[ $# -ne 1 ]]; then
  print_usage
  exit 1
fi

BENCHMARK_NAME="$1"

echo "Starting omni for $BENCHMARK_NAME benchmark..."
./start_omni.sh "$BENCHMARK_NAME"

sleep 2

python3.8 make_request.py > response.json

sleep 2

echo "Stopping omni for $BENCHMARK_NAME benchmark..."
./stop_omni.sh
