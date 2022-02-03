import threading
import uuid
from concurrent import futures
from logging import info, error
import asyncio
from copy import copy, deepcopy

import grpc

import data_locality
import validation_pb2
import validation_pb2_grpc
import socket

LOCAL_TESTING = False


class WorkerJobMetadata:

    def __init__(self, job_id, worker_ref):
        self.job_id = job_id
        self.worker = worker_ref
        self.gis_joins = []
        self.status = "NEW"

    def complete(self):
        self.status = "DONE"

    def __repr__(self):
        return f"WorkerJobMetadata: job_id={self.job_id}, worker={self.worker.hostname}, status={self.status}, gis_joins={self.gis_joins}"


class WorkerMetadata:

    def __init__(self, hostname, port):
        self.hostname = hostname
        self.port = port
        self.jobs = {}  # Mapping of { job_id -> WorkerJobMetadata }
        self.gis_joins = []  # List of gis_joins local to this worker

    def add_gis_join(self, gis_join):
        self.gis_joins.append(gis_join)

    def __repr__(self):
        return f"WorkerMetadata: hostname={self.hostname}, port={self.port}, jobs={self.jobs}, gis_joins={self.gis_joins}"


def generate_job_id():
    return uuid.uuid4().hex


def get_or_create_worker_job(worker, job_id):
    info(f"Creating job for worker={worker.hostname}, job={job_id}")
    if job_id not in worker.jobs:
        worker.jobs[job_id] = WorkerJobMetadata(job_id, worker)
    return worker.jobs[job_id]


class Master(validation_pb2_grpc.MasterServicer):

    def __init__(self, gis_join_locations):
        super(Master, self).__init__()
        self.tracked_workers = {}  # Mapping of { hostname -> WorkerMetadata }
        self.tracked_jobs = []  # List of JobMetadata
        self.saved_models_path = "testing/master/saved_models"
        self.gis_join_locations = gis_join_locations  # Mapping of { gis_join -> [hostname_1, hostname_2, hostname_3] }

    def is_worker_registered(self, hostname):
        return hostname in self.tracked_workers

    def RegisterWorker(self, request, context):
        info(f"Received WorkerRegistrationRequest: hostname={request.hostname}, port={request.port}")

        worker = WorkerMetadata(request.hostname, request.port)

        for gis_join, servers in self.gis_join_locations.items():
            for server in servers:
                if server == request.hostname:
                    worker.gis_joins.append(gis_join)

        self.tracked_workers[request.hostname] = worker
        info(f"Added Worker: {worker}")
        return validation_pb2.WorkerRegistrationResponse(success=True)

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
        #         # Upload file to all workers
        #         for worker in self.tracked_workers:
        #             with open(f"{self.saved_models_path}/{file_id}", "rb") as f:
        #                 with grpc.insecure_channel(f"{worker.hostname}:{worker.port}") as channel:
        #                     stub = validation_pb2_grpc.WorkerStub(channel)
        #                     upload_response = stub.UploadFile(file_chunker.chunk_file(f, file_id[:-4]))
        #                     if upload_response == validation_pb2.UPLOAD_STATUS_CODE_FAILED:
        #                         raise ValueError(f"Failed to upload file {file_id} to worker {worker.hostname}")
        #             info(f"Successfully uploaded file to worker {worker.hostname}")
        #
        #         # Success
        #         return validation_pb2.UploadStatus(
        #             message="Success",
        #             file_hash=hasher.hexdigest(),
        #             upload_status_code=validation_pb2.UPLOAD_STATUS_CODE_OK
        #         )
        #
        # except ValueError as e:
        #     error(f"{e}")
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

    # TODO: Handle concurrent responses, return to client
    # TODO: Test model file distribution as single request
    def SubmitValidationJob(self, request, context):
        # info(f"SubmitValidationJob Request: {request}")
        threads = []
        worker_jobs = []  # List of WorkerJobMetadata, which also contain a reference to a WorkerMetadata
        validation_job_responses = []
        job_id = generate_job_id()  # Random UUID for the job
        info(f"Generated job id {job_id}")
        info(f"GISJOINS: {request.gis_joins}")

        for gis_join in request.gis_joins:
            info(f"Processing job GISJOIN {gis_join}")
            worker_for_gis_join = None
            gis_join_hosts = self.gis_join_locations[gis_join]
            info(f"GISJOIN hosts: {gis_join_hosts}")
            for worker_host in gis_join_hosts:
                if self.is_worker_registered(worker_host):
                    worker_for_gis_join = self.tracked_workers[worker_host]
                    break

            if worker_for_gis_join is None:
                error(f"Unable to find a registered worker for GISJOIN {gis_join}")
                continue  # Skip this GISJOIN, there's no local workers for it

            # Found a registered worker for this GISJOIN, get or create a job for it
            worker_jobs.append(get_or_create_worker_job(worker_for_gis_join, job_id))

        # to be executed in a new thread
        def start_worker_thread(_worker: WorkerMetadata, _gis_joins_list):
            info(f"Submitting validation job to {_worker} for GISJOINs {_worker.gis_joins}")
            with grpc.insecure_channel(f"{_worker.hostname}:{_worker.port}") as channel:
                stub = validation_pb2_grpc.WorkerStub(channel)
                request.id = job_id
                request.gis_joins = _gis_joins_list
                validation_job_response = stub.BeginValidationJob(request)
                info(validation_job_response)
                validation_job_responses.append(validation_job_response)

        # Define async function to launch worker job
        async def run_worker_job(_worker_job: WorkerJobMetadata) -> None:
            info("Inside async run_worker_job()...")
            _worker = _worker_job.worker
            async with grpc.aio.insecure_channel(f"{_worker.hostname}:{_worker.port}") as channel:
                stub = validation_pb2_grpc.WorkerStub(channel)
                response = await stub.BeginValidationJob(validation_pb2.ValidationJobRequest(
                    id=_worker_job.job_id,
                    model_framework=request.model_framework,
                    model_type=request.model_type,
                    database=request.database,
                    collection=request.collection,
                    spatial_field=request.spatial_field,
                    label_field=request.label_field,
                    validation_metric=request.validation_metric,
                    feature_fields=request.feature_fields,
                    gis_joins=_worker_job.gis_joins,
                    model_file=request.model_file
                ))
            info(f"Response received: {response}")

        # Iterate over all the worker jobs created for this job, and launch them asynchronously
        for worker_job in worker_jobs:
            if len(worker_job.gis_joins) > 0:
                # start new thread
                # t = threading.Thread(target=start_worker_thread, args=(worker, gis_joins_list))
                # threads.append(t)
                # t.start()
                info(f"Launching worker job {worker_job}")
                # asyncio.run(run_worker_job(worker_job))


        # wait for all worker threads to complete
        # for thread in threads:
        #     thread.join()

        # TODO: combine results

        return validation_pb2.ValidationJobResponse(message="OK")


def run(master_port=50051):
    if LOCAL_TESTING:
        gis_join_locations = {}
    else:
        gis_join_locations = data_locality.get_gis_join_locations()

    # Initialize server and master
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    master = Master(gis_join_locations)
    validation_pb2_grpc.add_MasterServicer_to_server(master, server)
    hostname = socket.gethostname()

    # Start the server
    info(f"Starting master server on {hostname}:{master_port}")
    server.add_insecure_port(f"{hostname}:{master_port}")
    server.start()
    server.wait_for_termination()
