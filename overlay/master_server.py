import uuid
import asyncio
import socket
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from logging import info, error

import grpc

from overlay import validation_pb2
from overlay import validation_pb2_grpc
from overlay.db import shards, locality
from overlay.validation_pb2 import WorkerRegistrationResponse, ValidationJobResponse, ValidationJobRequest


class JobMetadata:

    def __init__(self, job_id, gis_joins):
        self.job_id = job_id
        self.worker_jobs = {}  # Mapping of { worker_hostname -> WorkerJobMetadata }
        self.gis_joins = gis_joins
        self.status = "NEW"


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

    def __init__(self, hostname, port, shard):
        self.hostname = hostname
        self.port = port
        self.shard = shard
        self.jobs = {}  # Mapping of { job_id -> WorkerJobMetadata }

    def __repr__(self):
        return f"WorkerMetadata: hostname={self.hostname}, port={self.port}, shard={self.shard.shard_name}, jobs={self.jobs}"


def generate_job_id() -> str:
    return uuid.uuid4().hex


def get_or_create_worker_job(worker, job_id):
    if job_id not in worker.jobs:
        info(f"Creating job for worker={worker.hostname}, job={job_id}")
        worker.jobs[job_id] = WorkerJobMetadata(job_id, worker)
    return worker.jobs[job_id]


def launch_worker_jobs_synchronously(job: JobMetadata, request: ValidationJobRequest) -> list:
    responses = []

    # Iterate over all the worker jobs created for this job and launch them serially
    for worker_hostname, worker_job in job.worker_jobs.items():
        if len(worker_job.gis_joins) > 0:
            info("Launching async run_worker_job()...")
            worker = worker_job.worker
            with grpc.insecure_channel(f"{worker.hostname}:{worker.port}") as channel:
                stub = validation_pb2_grpc.WorkerStub(channel)
                response = stub.BeginValidationJob(ValidationJobRequest(
                    id=worker_job.job_id,
                    model_framework=request.model_framework,
                    model_type=request.model_type,
                    database=request.database,
                    collection=request.collection,
                    gis_join_key=request.gis_join_key,
                    feature_fields=request.feature_fields,
                    label_field=request.label_field,
                    normalize_inputs=request.normalize_inputs,
                    validation_metric=request.validation_metric,
                    gis_joins=worker_job.gis_joins,
                    model_file=request.model_file
                ))
                responses.append(response)

    return responses


def launch_worker_jobs_concurrently(job: JobMetadata, request: ValidationJobRequest) -> list:
    responses = []

    # Define worker job function to be run in the thread pool
    def run_worker_job(_worker_job: WorkerJobMetadata, _request: ValidationJobRequest, _job_results: list) \
            -> ValidationJobResponse:
        info("Launching run_worker_job()...")
        _worker = _worker_job.worker
        with grpc.insecure_channel(f"{_worker.hostname}:{_worker.port}") as channel:
            stub = validation_pb2_grpc.WorkerStub(channel)
            response: ValidationJobResponse = stub.BeginValidationJob(ValidationJobRequest(
                id=_worker_job.job_id,
                model_framework=_request.model_framework,
                model_type=_request.model_type,
                database=_request.database,
                collection=_request.collection,
                gis_join_key=_request.gis_join_key,
                feature_fields=_request.feature_fields,
                label_field=_request.label_field,
                normalize_inputs=_request.normalize_inputs,
                validation_metric=_request.validation_metric,
                gis_joins=_worker_job.gis_joins,
                model_file=_request.model_file
            ))
            return response

    # Iterate over all the worker jobs created for this job and submit them to the thread pool executor
    executors_list = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        for worker_hostname, worker_job in job.worker_jobs.items():
            if len(worker_job.gis_joins) > 0:
                executors_list.append(executor.submit(run_worker_job, worker_job, request))

    # Wait on all tasks to finish -- Iterate over completed tasks, get their result, and log/append to responses
    for future in as_completed(executors_list):
        info(future)
        responses.append(future.result())

    return responses


async def launch_worker_jobs(job: JobMetadata, request: ValidationJobRequest, job_results: list):
    jobs_to_launch = []

    # Define async function to launch worker job
    async def run_worker_job(_worker_job: WorkerJobMetadata, _request: ValidationJobRequest,
                             _job_results: list) -> None:
        info("Launching async run_worker_job()...")
        _worker = _worker_job.worker
        async with grpc.aio.insecure_channel(f"{_worker.hostname}:{_worker.port}") as channel:
            stub = validation_pb2_grpc.WorkerStub(channel)
            response = await stub.BeginValidationJob(ValidationJobRequest(
                id=_worker_job.job_id,
                model_framework=_request.model_framework,
                model_type=_request.model_type,
                database=_request.database,
                collection=_request.collection,
                gis_join_key=_request.gis_join_key,
                feature_fields=_request.feature_fields,
                label_field=_request.label_field,
                normalize_inputs=_request.normalize_inputs,
                validation_metric=_request.validation_metric,
                gis_joins=_worker_job.gis_joins,
                model_file=_request.model_file
            ))
            info(f"Response received: {response}")
            _job_results.append(response)

    # Iterate over all the worker jobs created for this job and create asyncio jobs for them
    for worker_hostname, worker_job in job.worker_jobs.items():
        if len(worker_job.gis_joins) > 0:
            jobs_to_launch.append(asyncio.create_task(run_worker_job(worker_job, request, job_results)))

    await asyncio.gather(*jobs_to_launch)


class Master(validation_pb2_grpc.MasterServicer):

    def __init__(self, gis_join_locations, shard_metadata, local_testing=False):
        super(Master, self).__init__()
        self.tracked_workers = {}  # Mapping of { hostname -> WorkerMetadata }
        self.tracked_jobs = []  # List of JobMetadata
        self.saved_models_path = "testing/master/saved_models"
        self.gis_join_locations = gis_join_locations  # Mapping of { gis_join -> ShardMetadata }
        self.shard_metadata = shard_metadata
        self.local_testing = local_testing

    def is_worker_registered(self, hostname):
        return hostname in self.tracked_workers

    def choose_worker_from_shard(self, shard):
        for worker_host in shard.shard_servers:
            if self.is_worker_registered(worker_host):
                return self.tracked_workers[worker_host]
        return None

    def RegisterWorker(self, request, context):
        info(f"Received WorkerRegistrationRequest: hostname={request.hostname}, port={request.port}")

        for shard in self.shard_metadata.values():
            for shard_server in shard.shard_servers:
                if shard_server == request.hostname:
                    worker = WorkerMetadata(request.hostname, request.port, shard)
                    info(f"Successfully added Worker: {worker}, responsible for GISJOINs {shard.gis_joins}")
                    self.tracked_workers[request.hostname] = worker
                    return WorkerRegistrationResponse(success=True)

        error(f"Failed to find a shard that {request.hostname} belongs to")
        return WorkerRegistrationResponse(success=False)

    def DeregisterWorker(self, request, context):
        info(f"Received Worker(De)RegistrationRequest: hostname={request.hostname}, port={request.port}")

        if self.is_worker_registered(request.hostname):
            info(f"Worker {request.hostname} is registered. Removing...")
            del self.tracked_workers[request.hostname]
            info(f"Worker {request.hostname} is now deregistered and removed.")
            return WorkerRegistrationResponse(success=True)
        else:
            error(f"Worker {request.hostname} is not registered, can't remove")
            return WorkerRegistrationResponse(success=False)

    def SubmitValidationJob(self, request, context):
        info(f"SubmitValidationJob request for GISJOINs {request.gis_joins}")
        job_id = generate_job_id()  # Random UUID for the job
        job = JobMetadata(job_id, request.gis_joins)
        info(f"Created job id {job_id}")

        for gis_join in request.gis_joins:
            shard_hosting_gis_join = self.gis_join_locations[gis_join]
            worker = self.choose_worker_from_shard(shard_hosting_gis_join)
            if worker is None:
                error(f"Unable to find registered worker for GISJOIN {gis_join}")
                continue

            # Found a registered worker for this GISJOIN, get or create a job for it, and update jobs map
            worker_job = get_or_create_worker_job(worker, job_id)
            worker_job.gis_joins.append(gis_join)
            job.worker_jobs[worker.hostname] = worker_job

        for worker_hostname, worker_job in job.worker_jobs.items():
            info(f"{worker_hostname}: {worker_job}")

        # Submit jobs with asyncio and collect results
        # validation_job_responses = launch_worker_jobs_synchronously(job, request)
        validation_job_responses = launch_worker_jobs_concurrently(job, request)

        # asyncio.run(launch_worker_jobs(job, request, validation_job_responses))
        #for response in validation_job_responses:
        #    info(f"Response: {response}")

        return ValidationJobResponse(id=job_id)


def run(master_port=50051, local_testing=False):

    # Emulated environment
    if local_testing:
        shard_metadata: dict = {
            "shard1rs": shards.ShardMetadata("shard1rs", [socket.gethostname()])
        }
        gis_join_locations: dict = {"G2000010": shard_metadata["shard1rs"]}

    # Production environment -- discover chunk/shard locations
    else:
        shard_metadata: dict = shards.discover_shards()
        if shard_metadata is None:
            error("Shard discovery returned None. Exiting...")
            exit(1)

        gis_join_locations: dict = locality.get_gis_join_chunk_locations(shard_metadata)
        for shard in shard_metadata.values():
            info(shard)

    # Initialize server and master
    server = grpc.server(ThreadPoolExecutor(max_workers=10))
    master = Master(gis_join_locations, shard_metadata, local_testing)
    validation_pb2_grpc.add_MasterServicer_to_server(master, server)
    hostname = socket.gethostname()

    # Start the server
    info(f"Starting master server on {hostname}:{master_port}")
    server.add_insecure_port(f"{hostname}:{master_port}")
    server.start()
    server.wait_for_termination()
