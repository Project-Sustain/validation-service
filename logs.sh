#!/bin/bash

function print_usage {
  echo -e "./logs.sh"
  echo -e "Follows the logs of the current validation service process"
}

[[ $1 == "--help" ]] && print_usage && exit 0

tail -n 200 -f log.txt