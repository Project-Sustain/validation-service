#!/bin/bash

function print_usage {
  echo -e "./stop_free.sh <name>"
  echo -e "EXAMPLE ./stop_free.sh sh13rs_gisjoins_vs"
}

[[ $# -eq 1 ]] || (print_usage; exit 1)

"$PROJECT_MANAGEMENT/cluster/monitoring/free/stop_all.sh" "$1"