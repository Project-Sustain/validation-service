from logging import info, error
from typing import Iterator

import grpc
import os
import io
import zipfile
import signal
from concurrent import futures
from loky import get_reusable_executor

import socket

from overlay import validation_pb2_grpc
from overlay.validation_pb2 import WorkerRegistrationRequest, WorkerRegistrationResponse, JobMode, ModelFramework, \
    ValidationJobRequest, WorkerValidationJobResponse, ModelFileType, GisJoinMetadata, Metric
from overlay.constants import DB_HOST, DB_PORT, DB_NAME, MODELS_DIR
from overlay.profiler import Timer
from overlay.validation.tensorflow import TensorflowValidator
from overlay.validation.scikitlearn import ScikitLearnValidator
from overlay.validation.pytorch import PyTorchValidator
from overlay.db.shards import get_rs_member_state
from overlay.db.locality import discover_gis_joins
from overlay.structures import GisTree

# Loky shared, reusable ProcessPoolExecutor
shared_executor = get_reusable_executor(max_workers=8, timeout=10)


class Worker(validation_pb2_grpc.WorkerServicer):

    def __init__(self, master_hostname: str, master_port: int, hostname: str, port: int):
        super(Worker, self).__init__()
        self.master_hostname: str = master_hostname
        self.master_port: int = master_port
        self.hostname: str = hostname
        self.port: int = port
        self.jobs: list = []
        self.saved_models_path: str = MODELS_DIR
        self.is_registered = False
        self.gis_tree: GisTree = GisTree()
        info("Made it into worker_server, just above discover_gis_joins()")
        self.local_gis_joins: dict = discover_gis_joins()  # { gis_join -> count }
        for gis_join, count in self.local_gis_joins.items():
            self.gis_tree.insert_county(gis_join, {"count": count})

        self.rs_name, self.rs_member = get_rs_member_state()  # shard7rs, ReplicaSetMembership.PRIMARY
        self.register()

    # Register ourselves with the master
    def register(self):
        with grpc.insecure_channel(f"{self.master_hostname}:{self.master_port}") as channel:
            # Create gRPC GisJoinMetadata objects from discovered GISJOIN counts
            gis_join_metadata = []
            for gis_join, count in self.local_gis_joins.items():
                gis_join_metadata.append(GisJoinMetadata(
                    gis_join=gis_join,
                    count=count
                ))

            stub = validation_pb2_grpc.MasterStub(channel)
            registration_response: WorkerRegistrationResponse = stub.RegisterWorker(
                WorkerRegistrationRequest(
                    hostname=self.hostname,
                    port=self.port,
                    rs_name=self.rs_name,
                    rs_member=self.rs_member,
                    local_gis_joins=gis_join_metadata)
            )

            if registration_response.success:
                self.is_registered = True
                info(f"Successfully registered worker {self.hostname}:{self.port}")
            else:
                error(f"Failed to register worker {self.hostname}:{self.port}: {registration_response}")

    def __repr__(self) -> str:
        return f"Worker: hostname={self.hostname}, port={self.port}, jobs={self.jobs}"

    def BeginValidationJob(self, request: ValidationJobRequest, context) -> Iterator[Metric]:

        # info(f"Worker::BeginValidationJob(): Received Request: {request}")
        info(f"Worker::BeginValidationJob(): Received Request:")
        info(f"model_file.type: {request.model_file.type}")
        info(f"model_file.md5_hash: {request.model_file.md5_hash}")
        info(f"model_file.data.length: {len(request.model_file.data)}")
        info(f"mongo_host: {request.mongo_host}")
        info(f"mongo_port: {request.mongo_port}")
        info(f"read_config: {request.read_config}")
        info(f"database: {request.database}")
        info(f"collection: {request.collection}")
        info(f"normalize_inputs: {request.normalize_inputs}")
        info(f"label_field: {request.label_field}")
        info(f"feature_fields: {request.feature_fields}")
        info(f"model_framework: {request.model_framework}")
        info(f"model_category: {request.model_category}")
        info(f"loss_function: {request.loss_function}")
        info(f"spatial_coverage: {request.spatial_coverage}")
        info(f"allocations: {request.allocations}")
        info("==========================================================")

        # Save model
        if not self.save_model(request):
            err_msg = f"Unable to save {str(request.model_framework)} model file with type " \
                      f"{str(request.model_file.type)}!"
            error(err_msg)
            return

        info(f"Worker::BeginValidationJob(): Model Framework: {ModelFramework.Name(request.model_framework)}")
        # Select model framework, then launch jobs
        if request.model_framework == ModelFramework.TENSORFLOW:

            tf_validator: TensorflowValidator = TensorflowValidator(request, shared_executor, self.local_gis_joins)
            for metric in tf_validator.validate_gis_joins():
                info(f"Worker::BeingValidationJob(): Yielding metric from: {metric}")
                yield metric

        elif request.model_framework == ModelFramework.SCIKIT_LEARN:

            skl_validator: ScikitLearnValidator = ScikitLearnValidator(request, shared_executor, self.local_gis_joins)
            return skl_validator.validate_gis_joins()

        elif request.model_framework == ModelFramework.PYTORCH:

            pytorch_validator: PyTorchValidator = PyTorchValidator(request, shared_executor, self.local_gis_joins)
            return pytorch_validator.validate_gis_joins()

        else:
            err_msg = f"Unsupported model framework type {ModelFramework.Name(request.model_framework)}"
            error(err_msg)
            return
        return

    def save_model(self, request: ValidationJobRequest) -> bool:
        ok = True

        # Make the directory
        model_dir = f"{self.saved_models_path}/{request.id}"
        os.mkdir(model_dir)
        info(f"Worker::save_model(): Created directory {model_dir}")

        file_extension = "pkl"  # Default for Scikit-Learn pickle type

        # Validate the file type with the framework
        if request.model_framework == ModelFramework.TENSORFLOW:

            # Saved Tensorflow models have to be either SavedModel or HDF5 format:
            # https://www.tensorflow.org/tutorials/keras/save_and_load#save_the_entire_model
            if request.model_file.type == ModelFileType.TENSORFLOW_SAVED_MODEL_ZIP:
                zip_file = zipfile.ZipFile(io.BytesIO(request.model_file.data))
                zip_file.extractall(model_dir)
                return ok
            elif request.model_file.type == ModelFileType.TENSORFLOW_HDF5:
                file_extension = "h5"
            else:
                return not ok

        elif request.model_framework == ModelFramework.SCIKIT_LEARN:

            # Saved Scikit-Learn models have to be in the Pickle format:
            # https://scikit-learn.org/stable/modules/model_persistence.html
            if request.model_file.type != ModelFileType.SCIKIT_LEARN_PICKLE:
                return not ok

        elif request.model_framework == ModelFramework.PYTORCH:

            # Saved PyTorch models have to be in the Pickle zipfile format:
            # https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-model-for-inference
            if request.model_file.type != ModelFileType.PYTORCH_TORCHSCRIPT:
                return not ok
            file_extension = "pt"

        # Save the model with appropriate extension
        model_file_path = f"{model_dir}/{request.id}.{file_extension}"
        with open(model_file_path, "wb") as binary_file:
            binary_file.write(request.model_file.data)

        info(f"Worker::save_model(): Finished saving model to {model_file_path}")
        return ok

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
            info("We are not registered, no need to deregister")


def shutdown_gracefully(worker: Worker) -> None:
    worker.deregister()
    shared_executor.shutdown()
    exit(0)


def make_models_dir_if_not_exists() -> None:
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)


def run(master_hostname="localhost", master_port=50051, worker_port=50055) -> None:
    if MODELS_DIR == "":
        error("MODELS_DIR environment variable must be set!")
        exit(1)

    make_models_dir_if_not_exists()

    info(f"Environment: DB_HOST={DB_HOST}, DB_PORT={DB_PORT}, DB_NAME={DB_NAME}, MODELS_DIR={MODELS_DIR}")

    # Initialize server and worker
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    worker: Worker = Worker(master_hostname, master_port, socket.gethostname(), worker_port)

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
