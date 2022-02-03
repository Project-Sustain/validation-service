#!/bin/bash

function print_usage {
  echo -e "./run_flask_server.sh <master_uri> <flask_port>"
  echo -e "EXAMPLE ./run_flask_server.sh lattice-100:50051 5000"
}

if [[ $# -eq 2 ]]; then

  MASTER_URI=$1
  FLASK_PORT=$2
  python3.8 overlay --flaskserver --master_uri="$MASTER_URI" --port="$FLASK_PORT"

else
  print_usage
fi
