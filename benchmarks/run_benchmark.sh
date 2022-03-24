#!/bin/bash

function print_usage {
  echo -e "./run_benchmark.sh <name>"
  echo -e "EXAMPLE ./run_benchmark.sh sh13rs_gisjoins_vs"
}

[[ $# -eq 1 ]] || print_usage && exit 1

BENCHMARK_NAME="$1"

echo "Starting omni for $BENCHMARK_NAME benchmark..."
./start_omni.sh "$BENCHMARK_NAME"

curl --location --request POST 'lattice-150.cs.colostate.edu:5000/validation_service/submit_validation_job' \
--form 'file=@"/s/parsons/b/others/sustain/SustainProject/validation-service/testing/test_models/tensorflow/linear_regression/hdf5/my_model.h5"' \
--form 'request=@"/s/parsons/b/others/sustain/SustainProject/validation-service/testing/test_requests/test_request_all_gis_joins.json"' > response.txt

echo "Stopping omni for $BENCHMARK_NAME benchmark..."
./stop_omni.sh