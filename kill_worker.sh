#!/bin/bash

WORKER_PROCESS=$(ps -aux | grep "[r]un_worker.sh")

if [[ $WORKER_PROCESS != "" ]]; then
  echo "$WORKER_PROCESS"
  WORKER_PID=$(echo "$WORKER_PROCESS" | awk '{ print $2 }')

  kill "$WORKER_PID"
  echo "Found and killed worker process with pid $WORKER_PID"
else
  echo "Did not find any worker processes to kill"
fi