#!/bin/bash

function print_usage {
  echo -e "./start_benchmark.sh <name>"
  echo -e "EXAMPLE ./start_benchmark.sh sh13rs_gisjoins_vs"
}

[[ $# -eq 1 ]] || (print_usage; exit 1)

BENCHMARK_NAME=$1
echo "$BENCHMARK_NAME" > ".benchmark_name"

MON_ID=$(omni start | grep "started monitor with id" | awk '{ print $6 }')
echo "$MON_ID" > ".mon_id"