#!/bin/bash

function print_usage {
  echo -e "./run_worker.sh <master_uri> <worker_port>"
  echo -e "EXAMPLE ./run_worker.sh lattice-100:50051 50055"
}

[ $# -lt 2 ] && print_usage && exit 1

MASTER_URI="$1"
WORKER_PORT="$2"
DAEMON="$3"

echo -e "DB_HOST=$HOSTNAME\nDB_PORT=27018\nDB_NAME=sustaindb\nMODELS_DIR=/tmp/validation-service/saved_models" > .env

if [ "$DAEMON" == "--daemon" ]; then
  echo "Running worker as daemon"
  nohup python3.8 -m overlay --worker --master_uri="$MASTER_URI" --port="$WORKER_PORT" > log.txt 2>&1 & disown
else
  python3.8 -m overlay --worker --master_uri="$MASTER_URI" --port="$WORKER_PORT"
fi
