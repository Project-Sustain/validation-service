#!/bin/bash

WORKER_PROCESS=$(ps -aux | grep "[r]un_worker.sh")

if [[ $WORKER_PROCESS != "" ]]; then
  WORKER_PID=$($WORKER_PROCESS | awk '{ print $2 }')
  echo "Found worker process with pid $WORKER_PID"

  kill "$WORKER_PID"
else
  echo "Did not find any worker processes to kill"
fi