#!/bin/bash

function print_usage {
  echo -e "./start_free.sh <name>"
  echo -e "EXAMPLE ./start_free.sh"
}

[[ $# -eq 0 ]] || (print_usage; exit 1)

"$PROJECT_MANAGEMENT/cluster/monitoring/free/start_all.sh"