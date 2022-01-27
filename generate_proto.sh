#!/bin/bash

python3 -m grpc_tools.protoc -I overlay --python_out=overlay --grpc_python_out=overlay overlay/validation.proto