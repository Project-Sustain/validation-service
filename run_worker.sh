#!/bin/bash

function print_usage {
  echo -e "./run_worker.sh <master_hostname> <worker_port>"
}

[ $# -ne 2 ] && print_usage && exit 1

MASTER_HOST="$1"
WORKER_PORT="$2"

python3 overlay --worker --master_host="$MASTER_HOST" --port="$WORKER_PORT"