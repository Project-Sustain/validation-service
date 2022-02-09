from logging import info, error
import grpc
import os
import io
import zipfile
import signal
from concurrent import futures

import socket

from overlay import validation_pb2
from overlay import validation_pb2_grpc
from overlay.validation_pb2 import WorkerRegistrationRequest, WorkerRegistrationResponse
from overlay.constants import DB_HOST, DB_PORT, DB_NAME, MODELS_DIR
from overlay.db.querier import Querier
from overlay.tensorflow_validation import validation as tf_validation


class Worker(validation_pb2_grpc.WorkerServicer):

    def __init__(self, master_hostname, master_port, hostname, port):
        super(Worker, self).__init__()
        self.master_hostname = master_hostname
        self.master_port = master_port
        self.hostname = hostname
        self.port = port
        self.jobs = []
        self.saved_models_path = MODELS_DIR
        self.querier = Querier(f"mongodb://{DB_HOST}:{DB_PORT}", DB_NAME)
        self.is_registered = False

        # Register ourselves with the master
        with grpc.insecure_channel(f"{master_hostname}:{master_port}") as channel:
            stub = validation_pb2_grpc.MasterStub(channel)
            registration_response: WorkerRegistrationResponse = stub.RegisterWorker(
                WorkerRegistrationRequest(hostname=hostname, port=port)
            )

            if registration_response.success:
                self.is_registered = True
                info(f"Successfully registered worker {self.hostname}:{self.port}")
            else:
                error(f"Failed to register worker {self.hostname}:{self.port}: {registration_response}")

    def __repr__(self):
        return f"Worker: hostname={self.hostname}, port={self.port}, jobs={self.jobs}"

    def BeginValidationJob(self, request, context):
        info(f"Received BeginValidationJob Request: {request}")
        model_dir = f"{self.saved_models_path}/{request.id}"
        os.mkdir(model_dir)
        zip_file = zipfile.ZipFile(io.BytesIO(request.model_file.data))
        zip_file.extractall(model_dir)

        metrics = []  # list of proto ValidationMetric objects

        for gis_join in request.gis_joins:
            documents = self.querier.spatial_query(
                request.collection, request.gis_join_key, gis_join, request.feature_fields, request.label_field
            )

            # Calculate loss of model
            loss = tf_validation.validate_model(
                f"{self.saved_models_path}",
                request.id,
                request.model_type,
                documents,
                request.feature_fields,
                request.label_field,
                request.validation_metric,
                True
            )

            metrics.append(validation_pb2.ValidationMetric(
                gis_join=gis_join,
                loss=loss
            ))

        return validation_pb2.ValidationJobResponse(
            id=request.id,
            metrics=metrics
        )

    def deregister(self):
        if self.is_registered:
            # Deregister Worker from the Master
            with grpc.insecure_channel(f"{self.master_hostname}:{self.master_port}") as channel:
                stub = validation_pb2_grpc.MasterStub(channel)
                registration_response: WorkerRegistrationResponse = stub.DeregisterWorker(
                    WorkerRegistrationRequest(hostname=self.hostname, port=self.port)
                )

                if registration_response.success:
                    info(f"Successfully deregistered worker: {registration_response}")
                    self.is_registered = False
                else:
                    error(f"Failed to deregister worker: {registration_response}")
        else:
            info("We were never registered, no need to deregister")


def shutdown_gracefully(worker: Worker) -> None:
    worker.deregister()


def make_models_dir_if_not_exists() -> None:
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)


def run(master_hostname="localhost", master_port=50051, worker_port=50055) -> None:

    if MODELS_DIR == "":
        error("MODELS_DIR environment variable must be set!")
        exit(1)

    make_models_dir_if_not_exists()

    # Initialize server and worker
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    worker = Worker(master_hostname, master_port, socket.gethostname(), worker_port)

    # Set up Ctrl-C signal handling
    def call_shutdown(signum, frame):
        shutdown_gracefully(worker)
    signal.signal(signal.SIGINT, call_shutdown)

    validation_pb2_grpc.add_WorkerServicer_to_server(worker, server)
    hostname = socket.gethostname()

    # Start the server
    info(f"Starting worker server on {hostname}:{worker_port}")
    server.add_insecure_port(f"{hostname}:{worker_port}")
    server.start()
    server.wait_for_termination()
