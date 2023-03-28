#!/bin/bash

function print_usage {
  echo -e "DESCRIPTION\n\tInstalls the Validation Service dependencies\n"
  echo -e "USAGE\n\t./install.sh"
}

[[ $# -gt 0 ]] && print_usage && exit 0

python3 -m pip install --user -r requirements.txt