#!/bin/bash

function print_usage {
  echo -e "./run_master.sh <master_port>"
  echo -e "If no master port is specified, defaults to 50051."
  echo -e "EXAMPLE ./run_master.sh 50051"
}

MASTER_PORT=50051

if [[ $# -eq 1 ]]; then

  [[ $1 == "-h " ]]  && print_usage && exit 0

  MASTER_PORT=$1

fi

python3 overlay --master --port="$MASTER_PORT" > log.txt &