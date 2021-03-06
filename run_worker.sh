#!/bin/bash

function print_usage {
  echo -e "./run_worker.sh <master_uri> <worker_port>"
  echo -e "EXAMPLE ./run_worker.sh lattice-100:50051 50055"
}

[ $# -lt 2 ] && print_usage && exit 1

MASTER_URI="$1"
WORKER_PORT="$2"
DAEMON="$3"

if [ "$DAEMON" == "--daemon" ]; then
  echo "Running worker as daemon"

  WORKER_PROCESSES=$(ps -aux | grep "[o]verlay --worker")
  [ "$WORKER_PROCESSES" != "" ] && echo -e "Found worker processes running!\n$WORKER_PROCESSES\nPlease kill first before starting." && exit 1

  nohup python3.8 -m overlay --worker --master_uri="$MASTER_URI" --port="$WORKER_PORT" > log.txt 2>&1 & disown
else
  python3.8 -m overlay --worker --master_uri="$MASTER_URI" --port="$WORKER_PORT"
fi
