#!/bin/bash

function print_usage {
  echo -e "./run_benchmark.sh <name> <json_request_file>"
  echo -e "EXAMPLE ./run_benchmark.sh sh13rs_gisjoins_vs ../testing/test_requests/test_request_all_gis_joins.json"
}

[[ $# -eq 1 ]] || (print_usage; exit 1)

BENCHMARK_NAME="$1"
REQUEST_FILE_NAME="$2"

[[ -f "$REQUEST_FILE_NAME" ]] || (echo "File does not exist"; exit 1)

echo "Starting omni for $BENCHMARK_NAME benchmark..."
./start_omni.sh "$BENCHMARK_NAME"

curl --location --request POST 'lattice-150.cs.colostate.edu:5000/validation_service/submit_validation_job' \
--form 'file=@"../testing/test_models/tensorflow/linear_regression/hdf5/my_model.h5"' \
--form "request=@${REQUEST_FILE_NAME}" > response.txt

echo "Stopping omni for $BENCHMARK_NAME benchmark..."
./stop_omni.sh