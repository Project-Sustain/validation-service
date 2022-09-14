#!/bin/bash

function print_usage {
  echo -e "DESCRIPTION\n\tInstalls the Validation Service dependencies and generates the .env file for this host.\n"
  echo -e "USAGE\n\t./install.sh"
}

[[ $# -gt 0 ]] && print_usage && exit 0

python3.8 -m pip install --user -r requirements.txt && echo -e "DB_HOST=$HOSTNAME\nDB_PORT=27018\nDB_NAME=sustaindb\nMODELS_DIR=/tmp/validation-service/saved_models" > .env;
./generate_proto.sh