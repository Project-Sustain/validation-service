import uuid
import asyncio
import socket
import json
import grpc
import numpy as np
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from logging import info, error

from overlay import validation_pb2_grpc
from overlay.db import shards, locality
from overlay.db.shards import ShardMetadata
from overlay.profiler import Timer
from overlay.validation_pb2 import WorkerRegistrationRequest, WorkerRegistrationResponse, ValidationJobResponse, \
    ValidationJobRequest, JobMode, BudgetType, ValidationBudget, IncrementalVarianceBudget, SpatialCoverage, \
    SpatialAllocation, SpatialResolution


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
                del request_copy.allocations[:]
                request_copy.allocations.extend(job.gis_joins)
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
            del request_copy.allocations[:]
            request_copy.allocations.extend(_worker_job.gis_joins)
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
            del request_copy.allocations[:]
            request_copy.allocations.extend(_worker_job.gis_joins)
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


def save_intermediate_response_data(total_budget: int, initial_allocation: int, initial_response_metrics: list) -> None:
    total_gis_joins = len(initial_response_metrics)
    budget_used = initial_allocation * total_gis_joins
    remaining_budget = total_budget - budget_used
    save_obj = {
        "total_budget": total_budget,
        "initial_allocation": initial_allocation,
        "budget_used": budget_used,
        "remaining_budget": remaining_budget,
        "initial_response_metrics": []
    }

    for metric in initial_response_metrics:
        save_obj["initial_response_metrics"].append(
            {
                "gis_join": metric.gis_join,
                "variance": metric.variance,
                "loss": metric.loss,
                "allocation": metric.allocation,
                "duration_sec": metric.duration_sec
            }
        )

    filename = "/s/parsons/b/others/sustain/local-disk/a/tmp/intermediate_response.json"
    with open(filename, "w") as f:
        json.dump(save_obj, f)

    info(f"Saved {filename}")


def save_optimal_allocations(allocations: dict) -> None:
    filename = "/s/parsons/b/others/sustain/local-disk/a/tmp/optimal_allocations.json"
    with open(filename, "w") as f:
        json.dump(allocations, f)

    info(f"Saved {filename}")


def save_numpy_array(numpy_array) -> None:
    filename = "/s/parsons/b/others/sustain/local-disk/a/tmp/numpy_array.json"
    with open(filename, "w") as f:
        json.dump(numpy_array.tolist(), f)

    info(f"Saved {filename}")


def save_gis_join_counts(counts) -> None:
    filename = "/s/parsons/b/others/sustain/local-disk/a/tmp/gis_join_counts.json"
    with open(filename, "w") as f:
        json.dump(counts, f)

    info(f"Saved {filename}")


class Master(validation_pb2_grpc.MasterServicer):

    def __init__(self, hostname: str, port: int):
        super(Master, self).__init__()
        self.hostname = hostname
        self.port = port
        self.saved_models_path = "testing/master/saved_models"

        # Data structures
        self.tracked_workers = {}       # Mapping of { hostname -> WorkerMetadata }
        self.tracked_jobs = []          # list(JobMetadata)
        self.gis_join_locations = {}    # Mapping of { gis_join -> ShardMetadata }
        self.gis_join_metadata = {}     # Mapping of { gis_join -> count }
        self.shard_metadata = {}        # Mapping of { shard_name -> ShardMetadata }

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
                    # info(f"Worker {worker_host} currently has no job, defaulting to it for next GISJOIN in job={job_id}")
                    return self.tracked_workers[worker_host]
                elif len(worker_job.gis_joins) < min_gis_joins:
                    min_gis_joins = len(worker_job.gis_joins)
                    selected_worker = self.tracked_workers[worker_host]

        # info(f"Selecting {selected_worker.hostname} for next GISJOIN in job={job_id}")
        return selected_worker

    # Generates a JobMetadata object from the set of GISJOIN allocations
    def create_job_from_allocations(self, spatial_allocations: list) -> JobMetadata:

        job_id: str = generate_job_id()  # Random UUID for the job
        job: JobMetadata = JobMetadata(job_id, [allocation.gis_join for allocation in spatial_allocations])
        info(f"Created job id {job_id}")

        # Find and select workers with GISJOINs local to them
        for spatial_allocation in spatial_allocations:
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

        return job

    def get_request_allocations(self, request: ValidationJobRequest, strata_limit: int, sample_rate: float) -> \
            (list, bool, str):  # Returns list(SpatialAllocation), ok, err_msg

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
                return [], False, "Cannot specify spatial_coverage=ALL but then supply a non-empty list of GISJOINs!"
        elif request.spatial_coverage == SpatialCoverage.SUBSET:
            for gis_join in request.gis_joins:
                spatial_allocations.append(SpatialAllocation(
                    gis_join=gis_join,
                    strata_limit=strata_limit,
                    sample_rate=sample_rate
                ))
        else:
            return [], False, f"Unsupported spatial_coverage type {SpatialCoverage.Name(request.spatial_coverage)}"

        return spatial_allocations, True, ""

    # Processes a job with an incremental variance validation budget.
    # Returns a list of WorkerValidationJobResponse objects.
    def process_job_with_variance_budget(self, request: ValidationJobRequest) -> (str, list):

        save_gis_join_counts(self.gis_join_metadata)

        # Establish initial allocations for request
        variance_budget: IncrementalVarianceBudget = request.validation_budget.variance_budget
        total_budget: int = variance_budget.total_budget
        initial_allocation: int = variance_budget.initial_allocation
        info(f"Establishing initial allocation of {initial_allocation} for {len(self.gis_join_locations)} GISJOINs")
        spatial_allocations, ok, err_msg = self.get_request_allocations(
            request, initial_allocation, 0.0
        )
        if not ok:
            error(err_msg)
            return "", []

        request.allocations.extend(spatial_allocations)

        # Create and launch a job from allocations, gather worker responses
        job: JobMetadata = self.create_job_from_allocations(spatial_allocations)
        worker_responses: list = launch_worker_jobs(request, job)  # list(WorkerValidationJobResponse)

        # Compute optimal allocations from remaining budget using Neyman Allocation
        all_gis_join_metrics: list = []  # list(ValidationMetric)
        all_gis_join_variances: list = []
        sum_of_all_variances: int = 0
        budget_used: int = 0
        for worker_response in worker_responses:
            for metric in worker_response.metrics:
                all_gis_join_metrics.append(metric)
                all_gis_join_variances.append(metric.variance)
                budget_used += metric.allocation  # True allocation, incase GISJOIN had < initial_allocation records
                sum_of_all_variances += metric.variance

        budget_left: int = total_budget - budget_used
        info(f"This leaves us with a leftover budget of {total_budget} - {budget_used} = {budget_left}")

        # Calculate mean of all variances
        mean_of_all_variances = sum_of_all_variances / len(all_gis_join_variances)
        info(f"Mean of all variances: {mean_of_all_variances}")

        # Calculate standard deviation of all variances
        variances_numpy = np.array(all_gis_join_variances)
        std_dev_all_variances = variances_numpy.std()
        info(f"Standard deviation of all variances: {std_dev_all_variances}")
        sorted_variances = np.sort(variances_numpy, axis=-1)[::-1]
        std_devs_away = (sorted_variances - mean_of_all_variances) / std_dev_all_variances
        info(f"Std devs away: {std_devs_away}")

        save_intermediate_response_data(total_budget, initial_allocation, all_gis_join_metrics)
        save_numpy_array(std_devs_away)

        filtered_gis_join_metrics: list = []
        sum_of_filtered_variances = 0.0
        for metric in all_gis_join_metrics:
            # If variance > 2 standard deviations above mean
            if (metric.variance - mean_of_all_variances) / std_dev_all_variances >= 2.0:
                filtered_gis_join_metrics.append(metric)
                sum_of_filtered_variances += metric.variance

        # Create list of new allocations
        new_allocations: list = []  # list(SpatialAllocations)
        allocation_stats = {}
        for metric in filtered_gis_join_metrics:

            # Neyman Allocation + initial allocation
            new_optimal_allocation = int((budget_left * metric.variance) / sum_of_filtered_variances)

            # Cap new allocation at size of GISJOIN; don't allocate more than that GISJOIN has
            if new_optimal_allocation > self.gis_join_metadata[metric.gis_join]:
                new_optimal_allocation = self.gis_join_metadata[metric.gis_join]

            allocation_stats[metric.gis_join] = {"initial": metric.allocation, "final": new_optimal_allocation}
            new_allocations.append(SpatialAllocation(
                gis_join=metric.gis_join,
                strata_limit=new_optimal_allocation,
                sample_rate=0.0
            ))

        # Replace total list of SpatialAllocations with new allocations
        del request.allocations[:]
        request.allocations.extend(new_allocations)

        # Create and launch 2nd job from allocations
        job: JobMetadata = self.create_job_from_allocations(new_allocations)
        new_worker_responses = launch_worker_jobs(request, job)
        for new_worker_response in new_worker_responses:
            for new_metric in new_worker_response:
                new_metric.iteration = 1
        worker_responses.extend(new_worker_responses)

        return job.job_id, worker_responses

    # Processes a job with either a default budget or static budget.
    # Returns a list of WorkerValidationJobResponse objects.
    def process_job_with_normal_budget(self, request: ValidationJobRequest) -> (str, list):

        # Defaults
        strata_limit = 0
        sample_rate = 0.0

        if request.validation_budget.budget_type == BudgetType.STATIC_BUDGET:
            static_budget = request.validation_budget.static_budget
            if 0.0 < static_budget.sample_rate <= 1.0:
                sample_rate = static_budget.sample_rate

            if static_budget.strata_limit > 0:
                strata_limit = static_budget.strata_limit

            # Choose an equal limit per GISJOIN/strata that sums to the total budget limit
            if static_budget.total_limit > 0:

                if request.spatial_coverage == SpatialCoverage.ALL:
                    requested_gis_join_count = len(self.gis_join_metadata)
                else:
                    requested_gis_join_count = len(request.gis_joins)

                if static_budget.total_limit > requested_gis_join_count:
                    strata_limit = static_budget.total_limit // requested_gis_join_count
                else:
                    info("Specified a total limit less than the number of GISJOINs. Defaulting to 1 per GISJOIN")
                    strata_limit = 1

        spatial_allocations, ok, err_msg = self.get_request_allocations(request, strata_limit, sample_rate)
        if not ok:
            error(err_msg)
            return "", []
        request.allocations.extend(spatial_allocations)

        # Create and launch a job from allocations
        job: JobMetadata = self.create_job_from_allocations(spatial_allocations)
        worker_responses: list = launch_worker_jobs(request, job)  # list(WorkerValidationJobResponse)

        # Aggregate to state level if requested
        #if request.spatial_resolution == SpatialResolution.STATE:

        return job.job_id, worker_responses

    # Registers a Worker, using the reported GisJoinMetadata objects to populate the known GISJOINs and counts
    # for the ShardMetadata objects.
    def RegisterWorker(self, request: WorkerRegistrationRequest, context):
        info(f"Received WorkerRegistrationRequest: hostname={request.hostname}, port={request.port}")

        # Create a ShardMetadata for the registered worker if its shard is not already known
        if request.rs_name not in self.shard_metadata:

            shard_gis_join_metadata = {}
            for local_gis_join in request.local_gis_joins:
                self.gis_join_metadata[local_gis_join.gis_join] = local_gis_join.count
                shard_gis_join_metadata[local_gis_join.gis_join] = local_gis_join.count

            self.shard_metadata[request.rs_name] = ShardMetadata(
                request.rs_name,
                [request.hostname],
                shard_gis_join_metadata
            )

            for local_gis_join in request.local_gis_joins:
                self.gis_join_locations[local_gis_join.gis_join] = self.shard_metadata[request.rs_name]

        # Otherwise, just add the registered worker's hostname to the known ShardMetadata object
        else:
            self.shard_metadata[request.rs_name].shard_servers.append(request.hostname)

        # Create a WorkerMetadata object for tracking
        shard: ShardMetadata = self.shard_metadata[request.rs_name]
        worker: WorkerMetadata = WorkerMetadata(request.hostname, request.port, shard)
        info(f"Successfully added Worker: {worker}, responsible for {len(shard.gis_join_metadata)} GISJOINs")
        self.tracked_workers[request.hostname] = worker
        return WorkerRegistrationResponse(success=True)

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

        # Process the job with either a variance budget or static/default budget
        if request.validation_budget.budget_type == BudgetType.INCREMENTAL_VARIANCE_BUDGET:
            job_id, worker_responses = self.process_job_with_variance_budget(request)

        else:  # Default or static budget
            job_id, worker_responses = self.process_job_with_normal_budget(request)

        errors = []
        ok = True

        if len(worker_responses) == 0:
            error_msg = "Did not receive any responses from workers"
            ok = False
            error(error_msg)
            errors.append(error_msg)
        else:
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


def run(master_port=50051):

    # Initialize server and master
    master_hostname: str = socket.gethostname()
    server = grpc.server(ThreadPoolExecutor(max_workers=10))
    master: Master = Master(master_hostname, master_port)
    validation_pb2_grpc.add_MasterServicer_to_server(master, server)

    # Start the server
    info(f"Starting master server on {master_hostname}:{master_port}")
    server.add_insecure_port(f"{master_hostname}:{master_port}")
    server.start()
    server.wait_for_termination()
