import logging
from logging import info
import grpc
import validation_pb2
import validation_pb2_grpc


def run(master_hostname="localhost", master_port=50051, worker_port=50055):
    logging.basicConfig(level=logging.INFO)
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = validation_pb2_grpc.MasterStub(channel)
        registration_response = stub.RegisterWorker(
            validation_pb2.WorkerRegistrationRequest(hostname="localhost", port=worker_port))
    info(f"Greeter response received: {registration_response}")
