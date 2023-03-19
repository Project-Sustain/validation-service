#!/bin/bash

function print_usage {
  echo -e "./run_master.sh <master_port>"
  echo -e "If no master port is specified, defaults to 50051."
  echo -e "EXAMPLE ./run_master.sh 50051"
}

MASTER_PORT=50051

MASTER_PROCESSES=$(ps -aux | grep "[o]verlay --master")
[ "$MASTER_PROCESSES" != "" ] && echo -e "Found master processes running!\n$MASTER_PROCESSES\nPlease kill first before starting." && exit 1

if [[ $# -ge 1 ]]; then
  [[ $1 == "-h" ]]  && print_usage && exit 0
  MASTER_PORT=$1
  if [ $# -eq 2 ] && [ "$2" == "--local" ]; then
    python3 -m overlay --master --port="$MASTER_PORT" --local
  else
    python3 -m overlay --master --port="$MASTER_PORT"
  fi
fi


