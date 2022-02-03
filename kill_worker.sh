#!/bin/bash

WORKER_PROCESS=$(ps -aux | grep "[o]verlay --worker")

if [[ $WORKER_PROCESS != "" ]]; then
  echo "$WORKER_PROCESS"

  kill $(ps -aux | grep "[o]verlay --worker" | awk '{ print $2 }')
else
  echo "Did not find any worker processes to kill"
fi