#!/bin/bash

python3.8 -m grpc_tools.protoc --proto_path="./overlay" --python_out="./overlay" --grpc_python_out="./overlay" overlay/validation.proto
sed -i 's/import validation_pb2 as validation__pb2/from overlay import validation_pb2 as validation__pb2/g' overlay/validation_pb2_grpc.py