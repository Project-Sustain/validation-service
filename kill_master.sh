#!/bin/bash

MASTER_PROCESS=$(ps -aux | grep "[o]verlay --master")

if [[ $MASTER_PROCESS != "" ]]; then
  echo "$MASTER_PROCESS"

  kill $(ps -aux | grep "[o]verlay --master" | awk '{ print $2 }')
else
  echo "Did not find any master processes to kill"
fi
