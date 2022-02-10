#!/bin/bash

WORKER_PROCESS=$(ps -aux | grep "[o]verlay --worker")

if [[ $WORKER_PROCESS != "" ]]; then
  echo "Killing with SIGINT: $WORKER_PROCESS"
  # Send SIGINT signal to worker process, shutting it down gracefully
  kill -INT $(ps -aux | grep "[o]verlay --worker" | awk '{ print $2 }')
else
  echo "Did not find any worker processes to kill"
fi