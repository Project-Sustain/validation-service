import uuid
import asyncio
import socket
import grpc
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from logging import info, error

from overlay import validation_pb2_grpc
from overlay.db import shards, locality
from overlay.db.shards import ShardMetadata
from overlay.profiler import Timer
from overlay.validation_pb2 import WorkerRegistrationRequest, WorkerRegistrationResponse, ValidationJobResponse, \
    ValidationJobRequest, JobMode, BudgetType, ValidationBudget, IncrementalVarianceBudget, SpatialCoverage, \
    SpatialAllocation


class JobMetadata:

    # Takes a job ID and a list of SpatialAllocations
    # Example: [
    #           {gis_join: "G5600050", strata_limit: 2000, sample_rate: 0.0},
    #           {gis_join: "G5600170", strata_limit: 2300, sample_rate: 0.0},
    #           ...
    #          ]
    def __init__(self, job_id: str, gis_joins: list):
        self.job_id = job_id
        self.worker_jobs = {}  # Mapping of { worker_hostname -> WorkerJobMetadata }
        self.gis_joins = gis_joins  # list of SpatialAllocation objects
        self.status = "NEW"


class WorkerJobMetadata:

    def __init__(self, job_id, worker_ref):
        self.job_id = job_id
        self.worker = worker_ref
        self.gis_joins = []  # list of SpatialAllocation objects
        self.status = "NEW"

    def complete(self):
        self.status = "DONE"

    def __repr__(self):
        gis_joins_str = ""
        for gis_join in self.gis_joins:
            gis_joins_str += "  {gis_join=%s, strata_limit=%d, sample_rate=%.2f}\n" \
                             % (gis_join.gis_join, gis_join.strata_limit, gis_join.sample_rate)
        return f"WorkerJobMetadata: job_id={self.job_id}, worker={self.worker.hostname}, status={self.status}, " \
               f"gis_joins=[\n{gis_joins_str}\n]"


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


def get_worker_job(worker: WorkerMetadata, job_id: str):
    if job_id in worker.jobs:
        return worker.jobs[job_id]
    return None


def get_or_create_worker_job(worker: WorkerMetadata, job_id: str) -> WorkerJobMetadata:
    if job_id not in worker.jobs:
        info(f"Creating job for worker={worker.hostname}, job={job_id}")
        worker.jobs[job_id] = WorkerJobMetadata(job_id, worker)
    return worker.jobs[job_id]


# Returns list of WorkerValidationJobResponses
def launch_worker_jobs_synchronously(job: JobMetadata, request: ValidationJobRequest) -> list:
    responses = []

    # Iterate over all the worker jobs created for this job and launch them serially
    for worker_hostname, worker_job in job.worker_jobs.items():
        if len(worker_job.gis_joins) > 0:
            info("Launching async run_worker_job()...")
            worker = worker_job.worker
            with grpc.insecure_channel(f"{worker.hostname}:{worker.port}") as channel:
                stub = validation_pb2_grpc.WorkerStub(channel)
                request_copy = ValidationJobRequest()
                request_copy.CopyFrom(request)
                request_copy.gis_joins[:] = job.gis_joins
                request_copy.id = worker_job.job_id

                responses.append(stub.BeginValidationJob(request_copy))

    return responses


# Returns list of WorkerValidationJobResponses
def launch_worker_jobs_multithreaded(job: JobMetadata, request: ValidationJobRequest) -> list:
    responses = []

    # Define worker job function to be run in the thread pool
    def run_worker_job(_worker_job: WorkerJobMetadata, _request: ValidationJobRequest) -> ValidationJobResponse:
        info("Launching run_worker_job()...")
        _worker = _worker_job.worker
        with grpc.insecure_channel(f"{_worker.hostname}:{_worker.port}") as channel:
            stub = validation_pb2_grpc.WorkerStub(channel)
            request_copy = ValidationJobRequest()
            request_copy.CopyFrom(_request)
            request_copy.gis_joins[:] = _worker_job.gis_joins
            request_copy.id = _worker_job.job_id

            return stub.BeginValidationJob(request_copy)

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


# Returns list of WorkerValidationJobResponses
def launch_worker_jobs_asynchronously(job: JobMetadata, request: ValidationJobRequest) -> list:

    # Define async function to launch worker job
    async def run_worker_job(_worker_job: WorkerJobMetadata, _request: ValidationJobRequest) -> ValidationJobResponse:
        info("Launching async run_worker_job()...")
        _worker = _worker_job.worker
        async with grpc.aio.insecure_channel(f"{_worker.hostname}:{_worker.port}") as channel:
            stub = validation_pb2_grpc.WorkerStub(channel)
            request_copy = ValidationJobRequest()
            request_copy.CopyFrom(_request)
            request_copy.gis_joins[:] = _worker_job.gis_joins
            request_copy.id = _worker_job.job_id

            return await stub.BeginValidationJob(request_copy)

    # Iterate over all the worker jobs created for this job and create asyncio tasks for them
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    tasks = []
    for worker_hostname, worker_job in job.worker_jobs.items():
        if len(worker_job.gis_joins) > 0:
            tasks.append(loop.create_task(run_worker_job(worker_job, request)))

    task_group = asyncio.gather(*tasks)
    responses = loop.run_until_complete(task_group)
    loop.close()

    return list(responses)


class Master(validation_pb2_grpc.MasterServicer):

    def __init__(self, gis_join_locations: dict, shard_metadata: dict, local_testing=False):
        super(Master, self).__init__()
        self.tracked_workers = {}  # Mapping of { hostname -> WorkerMetadata }
        self.tracked_jobs = []  # List of JobMetadata
        self.saved_models_path = "testing/master/saved_models"
        self.gis_join_locations = gis_join_locations  # Mapping of { gis_join -> ShardMetadata }
        self.shard_metadata = shard_metadata
        self.local_testing = local_testing

    def is_worker_registered(self, hostname):
        return hostname in self.tracked_workers

    def choose_worker_from_shard(self, shard: ShardMetadata, job_id: str) -> WorkerMetadata:
        # Find registered hostname in shard with the least current jobs
        min_gis_joins = 9999999
        selected_worker = None
        for worker_host in shard.shard_servers:
            if self.is_worker_registered(worker_host):
                worker_job = get_worker_job(self.tracked_workers[worker_host], job_id)
                if worker_job is None:
                    info(f"Worker {worker_host} currently has no job, defaulting to it for next GISJOIN in job={job_id}")
                    return self.tracked_workers[worker_host]
                elif len(worker_job.gis_joins) < min_gis_joins:
                    min_gis_joins = len(worker_job.gis_joins)
                    selected_worker = self.tracked_workers[worker_host]

        info(f"Selecting {selected_worker.hostname} for next GISJOIN in job={job_id}")
        return selected_worker

    def set_request_allocations(self, request: ValidationJobRequest) -> (bool, str):  # ok
        strata_limit, sample_rate = get_strata_limit_and_sample_rate(request)
        spatial_allocations: list = []  # list(SpatialAllocation)

        if request.spatial_coverage == SpatialCoverage.ALL:
            if len(request.gis_joins) == 0:
                for gis_join in self.gis_join_locations.keys():
                    spatial_allocations.append(SpatialAllocation(
                        gis_join=gis_join,
                        strata_limit=strata_limit,
                        sample_rate=sample_rate
                    ))
            else:
                return False, "Cannot specify spatial_coverage=ALL but then supply a non-empty list of GISJOINs!"
        elif request.spatial_coverage == SpatialCoverage.SUBSET:
            for gis_join in request.gis_joins:
                spatial_allocations.append(SpatialAllocation(
                    gis_join=gis_join,
                    strata_limit=strata_limit,
                    sample_rate=sample_rate
                ))
        else:
            return False, f"Unsupported spatial_coverage type {SpatialCoverage.Name(request.spatial_coverage)}"

        request.allocations.extend(spatial_allocations)
        return True, ""

    def RegisterWorker(self, request: WorkerRegistrationRequest, context):
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

    def SubmitValidationJob(self, request: ValidationJobRequest, context):

        # Time the entire job from start to finish
        profiler: Timer = Timer()
        profiler.start()

        if request.spatial_coverage == SpatialCoverage.ALL:
            info(f"SubmitValidationJob request for ALL {len(self.gis_join_locations)} GISJOINs")
        else:
            info(f"SubmitValidationJob request for {len(request.gis_joins)} GISJOINs")

        job_id: str = generate_job_id()  # Random UUID for the job
        job: JobMetadata = JobMetadata(job_id, request.gis_joins)
        info(f"Created job id {job_id}")

        # Sets up all the initial allocations per GISJOIN
        if request.validation_budget.budget_type == BudgetType.INCREMENTAL_VARIANCE_BUDGET:
            pass

        else:  # Default or static budget

            self.set_request_allocations(request)

            # Find and select workers with GISJOINs local to them
            for spatial_allocation in request.spatial_allocations:
                gis_join = spatial_allocation.gis_join
                shard_hosting_gis_join: ShardMetadata = self.gis_join_locations[gis_join]
                worker: WorkerMetadata = self.choose_worker_from_shard(shard_hosting_gis_join, job_id)
                if worker is None:
                    error(f"Unable to find registered worker for GISJOIN {gis_join}")
                    continue

                # Found a registered worker for this GISJOIN, get or create a job for it, and update jobs map
                worker_job: WorkerJobMetadata = get_or_create_worker_job(worker, job_id)
                worker_job.gis_joins.append(spatial_allocation)
                job.worker_jobs[worker.hostname] = worker_job

            # Log selected workers
            for worker_hostname, worker_job in job.worker_jobs.items():
                info(f"{worker_hostname}: {worker_job}")

            return ValidationJobResponse(
                id="test",
                ok=True,
                error_msg="",
                duration_sec=0.0,
                worker_responses=[]
            )

            # Gather all the WorkerValidationJobResponses and check for errors
            worker_responses = launch_worker_jobs(request, job)
            errors = []
            ok = True
            for worker_response in worker_responses:
                if not worker_response.ok:
                    ok = False
                    error_msg = f"{worker_response.hostname} error: {worker_response.error_msg}"
                    errors.append(error_msg)

            error_msg = f"errors: {errors}"
            profiler.stop()

            return ValidationJobResponse(
                id=job_id,
                ok=ok,
                error_msg=error_msg,
                duration_sec=profiler.elapsed,
                worker_responses=worker_responses
            )


# Gets the limit and sample rate from the budget if supplied, and if not,
# defaults to no limit and no sample rate.
def get_strata_limit_and_sample_rate(request: ValidationJobRequest) -> (int, float):
    # Defaults
    strata_limit = 0
    sample_rate = 0.0

    if request.validation_budget.budget_type == BudgetType.STATIC_BUDGET:
        static_budget = request.validation_budget.static_budget
        if 0.0 < static_budget.sample_rate <= 1.0:
            sample_rate = static_budget.sample_rate

        if static_budget.strata_limit > 0:
            strata_limit = static_budget.strata_limit

        if static_budget.total_limit > 0:
            # Choose an equal limit per GISJOIN/strata that sums to the total budget limit
            if static_budget.total_limit > len(request.gis_joins):
                strata_limit = static_budget.total_limit // len(request.gis_joins)
            else:
                info("Specified a total limit less than the number of GISJOINs. Defaulting to 1 per GISJOIN")
                strata_limit = 1

    return strata_limit, sample_rate


# Processes a job with an incremental variance validation budget.
# Returns a list of WorkerValidationJobResponse objects.
def process_job_with_variance_budget(request: ValidationJobRequest, job: JobMetadata) -> list:
    variance_budget: IncrementalVarianceBudget = request.validation_budget.variance_budget

    # Establish initial allocations
    # gis_join_allocations = []
    # for gis_join in request.gis_joins:
    #     gis_join_allocations.append(Allocation(
    #         gis_join=gis_join,
    #         allocation=variance_budget.initial_allocation
    #     ))
    # variance_budget.allocations[:] = gis_join_allocations
    # request.variance_budget = variance_budget
    #
    # results_for_allocations = []
    # worker_responses = launch_worker_jobs(request, job)

    return launch_worker_jobs(request, job)


# Launches a round of worker jobs based on the master job mode selected.
# Returns a list of WorkerValidationJobResponse objects.
def launch_worker_jobs(request: ValidationJobRequest, job: JobMetadata) -> list:
    # Select strategy for submitting the job from the master
    if request.master_job_mode == JobMode.MULTITHREADED:
        info("Launching jobs in multi-threaded mode")
        worker_validation_job_responses = launch_worker_jobs_multithreaded(job, request)
    elif request.master_job_mode == JobMode.ASYNCHRONOUS or request.master_job_mode == JobMode.DEFAULT_JOB_MODE:
        info("Launching jobs in asynchronous mode")
        worker_validation_job_responses = launch_worker_jobs_asynchronously(job, request)
    else:
        info("Launching jobs in synchronous mode")
        worker_validation_job_responses = launch_worker_jobs_synchronously(job, request)

    info("Received all validation responses, returning...")
    return worker_validation_job_responses


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

        locality.discover_gis_join_counts()
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
