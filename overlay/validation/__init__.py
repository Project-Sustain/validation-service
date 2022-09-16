import os
import socket
from typing import Iterator
import psutil
from logging import info, error
from concurrent.futures import as_completed, ThreadPoolExecutor

from overlay.validation_pb2 import ValidationMetric, ValidationJobRequest, JobMode, LossFunction, Metric
from overlay.constants import MODELS_DIR


class Validator:

    def __init__(self, request: ValidationJobRequest, shared_executor, gis_join_counts):
        self.request: ValidationJobRequest = request
        self.model_path = self.get_model_path()
        self.shared_executor = shared_executor
        self.gis_join_counts = gis_join_counts  # { gis_join -> count }
        if request.model_category == "REGRESSION":
            info(f"Validator::__init__(): Selecting validate_regression_model()")
            self.validate_model_function = validate_regression_model
        elif request.model_category == "CLASSIFICATION":
            self.validate_model_function = validate_classification_model
        info(f"Validator::__init__(): Selecting validate_classification_model()")
        self.hostname = socket.gethostname()

    def get_model_path(self):
        model_path = f"{MODELS_DIR}/{self.request.id}/"
        first_entry = os.listdir(model_path)[0]
        model_path += first_entry
        return model_path

    def validate_gis_joins(self, verbose: bool = True) -> \
            Iterator[Metric]:  # This needs to be a generator, so it needs to yield the futures as they come in

        info(f"Validator::validate_gis_joins(): Launching validation job for {len(self.request.gis_joins)} GISJOINs")

        # Convert protobuf "repeated" field type to a python list
        feature_fields = []
        for feature_field in self.request.feature_fields:
            feature_fields.append(feature_field)

        # metrics = []  # list of protobuf ValidationMetric objects to return

        # Synchronous job mode
        if self.request.worker_job_mode == JobMode.SYNCHRONOUS:

            # Make requests serially using the gis_joins list in the request and the static strata limit/sample rate
            for spatial_allocation in self.request.allocations:
                gis_join: str = spatial_allocation.gis_join
                gis_join_count: int = self.gis_join_counts[gis_join]
                returned_gis_join, allocation, loss, variance, iteration, ok, error_msg, duration_sec = \
                    self.validate_model_function(
                        gis_join=gis_join,
                        gis_join_count=gis_join_count,
                        model_path=self.model_path,
                        feature_fields=feature_fields,
                        label_field=self.request.label_field,
                        loss_function=LossFunction.Name(self.request.loss_function),
                        mongo_host=self.request.mongo_host,
                        mongo_port=self.request.mongo_port,
                        read_preference=self.request.read_config.read_preference,
                        read_concern=self.request.read_config.read_concern,
                        database=self.request.database,
                        collection=self.request.collection,
                        limit=spatial_allocation.strata_limit,
                        sample_rate=spatial_allocation.sample_rate,
                        normalize_inputs=self.request.normalize_inputs,
                        verbose=verbose
                    )

                # TODO // implement streaming
                # metrics.append(ValidationMetric(
                #     gis_join=returned_gis_join,
                #     allocation=allocation,
                #     loss=loss,
                #     variance=variance,
                #     duration_sec=duration_sec,
                #     ok=ok,
                #     error_msg=error_msg
                # ))
                yield Metric(
                    gis_join=gis_join,
                    allocation=allocation,
                    loss=loss,
                    variance=variance,
                    duration_sec=duration_sec,
                    iteration=iteration,
                    ok=ok,
                    error_msg=error_msg,
                    hostname=self.hostname
                )

        # Job mode not single-threaded; use either the shared ProcessPoolExecutor or ThreadPoolExecutor
        else:
            # Create a child process object or thread object for each GISJOIN validation job
            executors_list: list = []

            info(f"Validator::validate_gis_joins(): JobMode: {JobMode.Name(self.request.worker_job_mode)}")
            if self.request.worker_job_mode == JobMode.MULTITHREADED:
                with ThreadPoolExecutor(max_workers=10) as executor:
                    for spatial_allocation in self.request.allocations:
                        gis_join: str = spatial_allocation.gis_join
                        gis_join_count: int = self.gis_join_counts[gis_join]
                        info(f"Launching validation job for GISJOIN {spatial_allocation.gis_join}")
                        executors_list.append(
                            executor.submit(
                                self.validate_model_function,
                                gis_join,
                                gis_join_count,
                                self.model_path,
                                feature_fields,
                                self.request.label_field,
                                LossFunction.Name(self.request.loss_function),
                                self.request.mongo_host,
                                self.request.mongo_port,
                                self.request.read_config.read_preference,
                                self.request.read_config.read_concern,
                                self.request.database,
                                self.request.collection,
                                spatial_allocation.strata_limit,
                                spatial_allocation.sample_rate,
                                self.request.normalize_inputs,
                                False  # don't log summaries on concurrent model
                            )
                        )
            else:  # JobMode.MULTIPROCESSING
                for spatial_allocation in self.request.allocations:
                    gis_join: str = spatial_allocation.gis_join
                    gis_join_count: int = self.gis_join_counts[gis_join]
                    info(f"Launching validation job for GISJOIN {spatial_allocation.gis_join}")
                    executors_list.append(
                        self.shared_executor.submit(
                            self.validate_model_function,
                            gis_join,
                            gis_join_count,
                            self.model_path,
                            feature_fields,
                            self.request.label_field,
                            LossFunction.Name(self.request.loss_function),
                            self.request.mongo_host,
                            self.request.mongo_port,
                            self.request.read_config.read_preference,
                            self.request.read_config.read_concern,
                            self.request.database,
                            self.request.collection,
                            spatial_allocation.strata_limit,
                            spatial_allocation.sample_rate,
                            self.request.normalize_inputs,
                            False  # don't log summaries on concurrent model
                        )
                    )

            # Wait on all tasks to finish -- Iterate over completed tasks, get their result, and log/append to responses
            for future in as_completed(executors_list):
                gis_join, allocation, loss, variance, iteration, ok, error_msg, duration_sec = future.result()
                # metrics.append(ValidationMetric(
                #     gis_join=gis_join,
                #     allocation=allocation,
                #     loss=loss,
                #     variance=variance,
                #     duration_sec=duration_sec,
                #     ok=ok,
                #     error_msg=error_msg
                # ))
                info(f"Yielding metric for gis_join={gis_join}")
                yield Metric(
                    gis_join=gis_join,
                    allocation=allocation,
                    loss=loss,
                    variance=variance,
                    duration_sec=duration_sec,
                    iteration=iteration,
                    ok=ok,
                    error_msg=error_msg,
                    hostname=self.hostname
                )

            # all tasks completed. kill all child processes
            # parent_pid = os.getpid()
            # parent = psutil.Process(parent_pid)
            # for child in parent.children(recursive=True):
            #     info(f"Terminating Child Process: {child}")
            #     child.kill()

        # info(f"metrics: {len(metrics)} responses")
        # return metrics
        # // needs to be yield instead of return


# Stub for function that needs to be implemented in concrete subclasses
def validate_regression_model(
        gis_join: str,
        gis_join_count: int,
        model_path: str,
        feature_fields: list,
        label_field: str,
        loss_function: str,
        mongo_host: str,
        mongo_port: int,
        read_preference: str,
        read_concern: str,
        database: str,
        collection: str,
        limit: int,
        sample_rate: float,
        normalize_inputs: bool,
        verbose: bool = True) -> (str, int, float, float, bool, str, float):
    # Returns the gis_join, allocation, loss, variance, ok status, error message, and duration
    raise NotImplementedError("validate_regression_model() is not implemented for abstract class Validator.")


# Stub for function that needs to be implemented in concrete subclasses
def validate_classification_model(
        gis_join: str,
        gis_join_count: int,
        model_path: str,
        feature_fields: list,
        label_field: str,
        loss_function: str,
        mongo_host: str,
        mongo_port: int,
        read_preference: str,
        read_concern: str,
        database: str,
        collection: str,
        limit: int,
        sample_rate: float,
        normalize_inputs: bool,
        verbose: bool = True) -> (str, int, float, float, bool, str, float):
    # Returns the gis_join, allocation, loss, variance, ok status, error message, and duration
    raise NotImplementedError("validate_classification_model() is not implemented for abstract class Validator.")
