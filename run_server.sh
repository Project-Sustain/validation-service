#!/bin/bash

echo -e "Checking dependencies..."
python3 -m pip list | grep "Flask" || python3 -m pip install --user -r requirements.txt

python3 flaskr