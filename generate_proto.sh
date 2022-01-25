#!/bin/bash

python3 -m grpc_tools.protoc -I overlay/proto/ --python_out=overlay/proto --grpc_python_out=overlay/proto overlay/proto/validation.proto