#!/bin/bash

function print_usage {
  echo -e "./run_worker.sh <master_uri> <worker_port>"
  echo -e "EXAMPLE ./run_worker.sh lattice-100:50051 50055"
}

[ $# -ne 2 ] && print_usage && exit 1

MASTER_URI="$1"
WORKER_PORT="$2"

python3 overlay --worker --master_uri="$MASTER_HOST" --port="$WORKER_PORT"