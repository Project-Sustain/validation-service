#!/bin/bash

WORKER_PROCESSES=$(ps -aux | grep "[o]verlay --worker")

if [[ $WORKER_PROCESSES != "" ]]; then
  echo "Killing with SIGINT: $WORKER_PROCESSES"
  # Send SIGINT signal to worker process, shutting it down gracefully
  pkill -f overlay
  kill $(ps -aux | grep -P "[l]oky" | awk '{ print $2 }')
else
  echo "Did not find any worker processes to kill"
fi
