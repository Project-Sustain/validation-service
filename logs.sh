#!/bin/bash

function print_usage {
  echo -e "./logs.sh"
  echo -e "Follows the logs of the current validation service process"
}

if [[ $# -eq 0 ]]; then
  tail -n 200 -f log.txt
else
  print_usage
fi