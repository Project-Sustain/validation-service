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
from overlay.validation_pb2 import WorkerRegistrationRequest, WorkerRegistrationResponse, JobMode, ModelFramework, \
    ValidationJobRequest, ValidationJobResponse, ModelFile, ModelFileType
from overlay.constants import DB_HOST, DB_PORT, DB_NAME, MODELS_DIR
from overlay.db.querier import Querier
from overlay.tensorflow_validation.validation import TensorflowValidator
from overlay.scikitlearn_validation.validation import ScikitLearnValidator
from overlay.pytorch_validation.validation import PyTorchValidator


class Worker(validation_pb2_grpc.WorkerServicer):

    def __init__(self, master_hostname, master_port, hostname, port):
        super(Worker, self).__init__()
        self.master_hostname = master_hostname
        self.master_port = master_port
        self.hostname = hostname
        self.port = port
        self.jobs = []
        self.saved_models_path = MODELS_DIR
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

    def BeginValidationJob(self, request: ValidationJobRequest, context):
        info(f"Received BeginValidationJob Request: {request}")
        info(f"BeginValidationJob: validation_budget={request.validation_budget}")
        metrics = None

        # Save model
        if not self.save_model(request):
            err_msg = f"Unable to save {str(request.model_framework)} model file with type " \
                      f"{str(request.model_file.type)}!"
            error(err_msg)
            return ValidationJobResponse(id=request.id, ok=False, err_msg=err_msg)

        # Select model framework, then job mode
        if request.model_framework == ModelFramework.TENSORFLOW:

            tf_validator: TensorflowValidator = TensorflowValidator(request)

            if request.worker_job_mode == JobMode.MULTITHREADED:
                metrics = tf_validator.validate_gis_joins_multithreaded(request.gis_joins)
            elif request.worker_job_mode == JobMode.SYNCHRONOUS:
                metrics = tf_validator.validate_gis_joins_synchronous(request.gis_joins)
            elif request.worker_job_mode == JobMode.MULTIPROCESSING:
                metrics = tf_validator.validate_gis_joins_multiprocessing(request.gis_joins)
            else:
                err_msg = f"{request.worker_job_mode} job mode not implemented for Tensorflow validation!"
                error(err_msg)
                return ValidationJobResponse(id=request.id, ok=False, err_msg=err_msg)

        elif request.model_framework == ModelFramework.SCIKIT_LEARN:

            skl_validator: ScikitLearnValidator = ScikitLearnValidator(request)

            if request.worker_job_mode == JobMode.SYNCHRONOUS:
                metrics = skl_validator.validate_gis_joins_synchronous(request.gis_joins)
            elif request.worker_job_mode == JobMode.MULTITHREADED:
                metrics = skl_validator.validate_gis_joins_multithreaded(request.gis_joins)
            else:
                err_msg = f"{request.worker_job_mode} job mode not implemented for Scikit-Learn validation!"
                error(err_msg)
                return ValidationJobResponse(id=request.id, ok=False, err_msg=err_msg)

        elif request.model_framework == ModelFramework.PYTORCH:

            pytorch_validator: PyTorchValidator = PyTorchValidator(request)

            if request.worker_job_mode == JobMode.SYNCHRONOUS:
                metrics = pytorch_validator.validate_gis_joins_synchronous(request.gis_joins)
            elif request.worker_job_mode == JobMode.MULTITHREADED:
                metrics = pytorch_validator.validate_gis_joins_multithreaded(request.gis_joins)
            else:
                err_msg = f"{request.worker_job_mode} job mode not implemented for PyTorch validation!"
                error(err_msg)
                return ValidationJobResponse(id=request.id, ok=False, err_msg=err_msg)

        # Create and return response from aggregated metrics
        return ValidationJobResponse(
            id=request.id,
            ok=True,
            metrics=metrics
        )

    def save_model(self, request: ValidationJobRequest) -> bool:
        ok = True

        # Make the directory
        model_dir = f"{self.saved_models_path}/{request.id}"
        os.mkdir(model_dir)
        info(f"Created directory {model_dir}")

        file_extension = "pickle"  # Default for Scikit-Learn pickle type

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
            if request.model_file.type != ModelFileType.PYTORCH_ZIPFILE:
                return not ok
            file_extension = "pth"

        # Save the model with appropriate extension
        model_file_path = f"{model_dir}/{request.id}.{file_extension}"
        with open(model_file_path, "wb") as binary_file:
            binary_file.write(request.model_file.data)

        info(f"Finished saving model to {model_file_path}")
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
