#!/bin/bash

FLASK_PROCESS=$(ps -aux | grep "[o]verlay --flaskserver")

if [[ $FLASK_PROCESS != "" ]]; then
  echo "$FLASK_PROCESS"
  kill $(ps -aux | grep "[o]verlay --flaskserver" | awk '{ print $2 }')
else
  echo "Did not find any flaskserver processes to kill"
fi
