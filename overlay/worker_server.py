import logging
from logging import info, error
import grpc
import os
import io
import socket
import validation_pb2
import validation_pb2_grpc
import hashlib
import zipfile
from concurrent import futures
import tensorflow_validation
import socket


class Worker(validation_pb2_grpc.WorkerServicer):

    def __init__(self, master_hostname, master_port, hostname, port):
        super(Worker, self).__init__()
        self.master_hostname = master_hostname
        self.master_port = master_port
        self.hostname = hostname
        self.port = port
        self.jobs = []
        self.saved_models_path = "testing/worker/saved_models"

        # Register ourselves with the master
        with grpc.insecure_channel(f"{master_hostname}:{master_port}") as channel:
            stub = validation_pb2_grpc.MasterStub(channel)
            registration_response = stub.RegisterWorker(
                validation_pb2.WorkerRegistrationRequest(hostname=hostname, port=port)
            )
            info(registration_response)

    def __repr__(self):
        return f"Worker: hostname={self.hostname}, port={self.port}, jobs={self.jobs}"

    def UploadFile(self, request_iterator, context):
        # info(f"Received UploadFile stream request, processing chunks...")
        # total_bytes_received = 0
        # chunk_index = 0
        #
        # try:
        #     file_pointer = None
        #     file_id = None
        #     for file_chunk in request_iterator:
        #         file_id = file_chunk.id
        #         if not file_pointer:
        #             file_pointer = open(f"{self.saved_models_path}/{file_id}", "wb")
        #
        #         file_bytes = file_chunk.data
        #         info(f"Length of chunk index {chunk_index}: {len(file_bytes)}")
        #         chunk_index += 1
        #         total_bytes_received += len(file_bytes)
        #         file_pointer.write(file_bytes)
        #
        #     file_pointer.close()
        #     info(f"Finished receiving chunks, {total_bytes_received} total bytes received")
        #
        #     # Get file hash
        #     if file_id:
        #         with open(f"{self.saved_models_path}/{file_id}", "rb") as f:
        #             hasher = hashlib.md5()
        #             buf = f.read()
        #             hasher.update(buf)
        #         info(f"Uploaded file hash: {hasher.hexdigest()}")
        #
        #         # Unzip file
        #         with zipfile.ZipFile(f"{self.saved_models_path}/{file_id}", "r") as zip_ref:
        #             directory = f"{self.saved_models_path}/{file_id[:-4]}"
        #             os.mkdir(directory)
        #             zip_ref.extractall(directory)
        #
        #         # Success
        #         return validation_pb2.UploadStatus(
        #             message="Success",
        #             file_hash=hasher.hexdigest(),
        #             upload_status_code=validation_pb2.UPLOAD_STATUS_CODE_OK
        #         )
        #
        # except Exception as e:
        #     error(f"Failed to receive chunk index {chunk_index}: {e}")
        #
        # # Failure, hopefully we don't make it here
        # return validation_pb2.UploadStatus(
        #     message="Failed",
        #     file_hash="None",
        #     upload_status_code=validation_pb2.UPLOAD_STATUS_CODE_FAILED
        # )
        pass

    def BeginValidationJob(self, request, context):
        info(f"Received BeginValidationJob Request: {request}")
        model_dir = f"{self.saved_models_path}/{request.id}"
        os.mkdir(model_dir)
        zip_file = zipfile.ZipFile(io.BytesIO(request.model_file.data))
        zip_file.extractall(model_dir)

        # for gis_join in request.gis_joins:
        #     tensorflow_validation.validate_model(
        #         request.id,
        #         f"{self.saved_models_path}",
        #         request.model_type,
        #         request.database,
        #         request.collection,
        #         request.label_field,
        #         request.validation_metric,
        #         request.feature_fields,
        #         request.gis_joins
        #     )
        return validation_pb2.WorkerJobResponse(request.id, "", validation_pb2.WORKER_JOB_STATUS_CODE_OK)


def run(master_hostname="localhost", master_port=50051, worker_port=50055):
    # Initialize server and worker
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    worker = Worker(master_hostname, master_port, socket.gethostname(), worker_port)
    validation_pb2_grpc.add_WorkerServicer_to_server(worker, server)
    hostname = socket.gethostname()

    # Start the server
    info(f"Starting worker server on {hostname}:{worker_port}")
    server.add_insecure_port(f"{hostname}:{worker_port}")
    server.start()
    server.wait_for_termination()
