#!/bin/bash

BENCHMARK_NAME=$1
export BENCHMARK_NAME

[[ -z "$BENCHMARK_NAME" ]] && echo "BENCHMARK_NAME not set!" && exit 1
[[ -z "$MON_ID" ]] && echo "MON_ID not set!" && exit 1

echo -e "Stopping BENCHMARK_NAME=$BENCHMARK_NAME with MON_ID=$MON_ID..."
omni stop $MON_ID

echo -e "Collecting..."
omni collect $MON_ID "./$BENCHMARK_NAME"