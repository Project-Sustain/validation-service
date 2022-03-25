import concurrent.futures
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from logging import info, warning, error
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

from overlay.validation_pb2 import ValidationMetric, ValidationJobRequest, BudgetType, StaticBudget, BudgetType, \
    JobMode, LossFunction
from overlay.db.querier import Querier
from overlay.constants import MODELS_DIR
from overlay.tensorflow_validation.validation import normalize_dataframe
from overlay.profiler import Timer


class PyTorchValidator:

    def __init__(self, request: ValidationJobRequest):
        self.request: ValidationJobRequest = request
        self.model_path = self.get_model_path()

    def get_model_path(self):
        model_path = f"{MODELS_DIR}/{self.request.id}/"
        first_entry = os.listdir(model_path)[0]
        model_path += first_entry
        return model_path

    def validate_gis_joins(self, verbose: bool = True) -> list:
        # Convert protobuf "repeated" field type to a python list
        feature_fields = []
        for feature_field in self.request.feature_fields:
            feature_fields.append(feature_field)

        # Capture validation budget parameters
        if self.request.validation_budget.budget_type == BudgetType.STATIC_BUDGET:
            budget: StaticBudget = self.request.validation_budget.static_budget
            strata_limit: int = budget.strata_limit
            sample_rate: float = budget.sample_rate
        elif self.request.validation_budget.budget_type == BudgetType.INCREMENTAL_VARIANCE_BUDGET:
            error("Incremental variance budgeting not yet implemented!")
            return []
        else:
            error(f"Unrecognized budget type '{self.request.validation_budget.budget_type}'")
            return []

        metrics = []  # list of protobuf ValidationMetric objects to return

        # Select job mode
        if self.request.worker_job_mode == JobMode.SYNCHRONOUS:

            # Make requests serially
            for gis_join in self.request.gis_joins:
                loss, ok, error_msg, duration_sec = validate_model(
                    gis_join=gis_join,
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
                    limit=strata_limit,
                    sample_rate=sample_rate,
                    normalize_inputs=self.request.normalize_inputs,
                    verbose=verbose
                )

                metrics.append(ValidationMetric(
                    gis_join=gis_join,
                    loss=loss,
                    duration_sec=duration_sec,
                    ok=ok,
                    error_msg=error_msg
                ))

        # Job mode not single-threaded; either multi-thread or multi-processed
        else:

            # Choose executor type
            executor_type = ProcessPoolExecutor if self.request.worker_job_mode == JobMode.MULTIPROCESSING \
                or self.request.worker_job_mode == JobMode.DEFAULT_JOB_MODE else ThreadPoolExecutor

            executors_list: list = []
            with executor_type(max_workers=8) as executor:

                # Create either a thread or child process object for each GISJOIN validation job
                for gis_join in self.request.gis_joins:
                    info(f"Launching validation job for GISJOIN {gis_join}")
                    executors_list.append(
                        executor.submit(
                            validate_model,
                            gis_join,
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
                            strata_limit,
                            sample_rate,
                            self.request.normalize_inputs,
                            verbose
                        )
                    )

            # Wait on all tasks to finish -- Iterate over completed tasks, get their result, and log/append to responses
            for future in as_completed(executors_list):
                loss, ok, error_msg, duration_sec = future.result()
                metrics.append(ValidationMetric(
                    gis_join=gis_join,
                    loss=loss,
                    duration_sec=duration_sec,
                    ok=ok,
                    error_msg=error_msg
                ))

        return metrics


def validate_model(
        gis_join: str,
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
        verbose: bool = True) -> (float, bool, str, float):

    profiler: Timer = Timer()
    profiler.start()

    # Load PyTorch model from disk (OS should cache in memory for future loads)
    model = torch.load(model_path)

    if verbose:
        model_description = f'{model}\nParameters:\n'
        for param in model.parameters():
            model_description = f'{param}\n'

        info(f"Model :{model_description}")

    querier = Querier(
        mongo_host=mongo_host,
        mongo_port=mongo_port,
        db_name=database,
        read_preference=read_preference,
        read_concern=read_concern
    )

    documents = querier.spatial_query(
        collection_name=collection,
        gis_join=gis_join,
        features=feature_fields,
        label=label_field,
        limit=limit,
        sample_rate=sample_rate
    )

    # Load MongoDB Documents into Pandas DataFrame
    features_df = pd.DataFrame(list(documents))
    querier.close()

    info(f"Loaded Pandas DataFrame from MongoDB of size {len(features_df.index)}")

    if len(features_df.index) == 0:
        error("DataFrame is empty! Returning -1.0 for loss")
        return -1.0

    # Normalize features, if requested
    if normalize_inputs:
        features_df = normalize_dataframe(features_df)
        info(f"Normalized Pandas DataFrame")

    # Pop the label column off into its own DataFrame
    label_df = features_df.pop(label_field)

    # evaluate model
    X = torch.from_numpy(features_df.values.astype(np.float32))
    y = torch.from_numpy(label_df.values.astype(np.float32))
    y = y.view(y.shape[0], 1)  # convert y to a column vector

    n_samples, n_features = X.shape
    info(f'n_samples: {n_samples}, n_features: {n_features}')

    if loss_function == "MEAN_ABSOLUTE_ERROR":
        criterion = nn.MSELoss()
    elif loss_function == "MEAN_SQUARED_ERROR":
        criterion = nn.L1Loss()
    elif loss_function == "ROOT_MEAN_SQUARED_ERROR":
        # TODO: implement RMSE manually, PyTorch does not have an inbuilt RMSE loss function
        criterion = nn.MSELoss()
    elif loss_function == "NEGATIVE_LOG_LIKELIHOOD_LOSS":
        criterion = nn.NLLLoss()
    elif loss_function == "CROSS_ENTROPY_LOSS":
        criterion = nn.CrossEntropyLoss()
    else:
        warning(f"PyTorch validation: Unknown loss function: {loss_function}")
        return -1.0

    y_predicted = model(X)
    loss = criterion(y_predicted, y)
    profiler.stop()
    info(f"Model validation results: {loss}")

    return loss, True, "", profiler.elapsed
