import concurrent.futures
import os
from math import sqrt

from logging import info, error
from concurrent.futures import as_completed, ThreadPoolExecutor

from overlay.validation_pb2 import ValidationMetric, ValidationJobRequest, JobMode, LossFunction

from overlay.profiler import Timer
from overlay.constants import MODELS_DIR


class TensorflowValidator:

    def __init__(self, request: ValidationJobRequest, shared_executor, gis_join_counts):
        self.request: ValidationJobRequest = request
        self.model_path = self.get_model_path()
        self.shared_executor = shared_executor
        self.gis_join_counts = gis_join_counts  # { gis_join -> count }

    def get_model_path(self):
        model_path = f"{MODELS_DIR}/{self.request.id}/"
        first_entry = os.listdir(model_path)[0]
        model_path += first_entry
        return model_path

    def validate_gis_joins(self, verbose: bool = True) -> list:

        info(f"Launching validation job for {len(self.request.gis_joins)} GISJOINs")

        # Convert protobuf "repeated" field type to a python list
        feature_fields = []
        for feature_field in self.request.feature_fields:
            feature_fields.append(feature_field)

        metrics = []  # list of protobuf ValidationMetric objects to return

        # Synchronous job mode
        if self.request.worker_job_mode == JobMode.SYNCHRONOUS:

            # Make requests serially using the gis_joins list in the request and the static strata limit/sample rate
            for spatial_allocation in self.request.allocations:
                gis_join: str = spatial_allocation.gis_join
                gis_join_count: int = self.gis_join_counts[gis_join]
                returned_gis_join, allocation, loss, variance, ok, error_msg, duration_sec = \
                    validate_model(
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

                metrics.append(ValidationMetric(
                    gis_join=returned_gis_join,
                    allocation=allocation,
                    loss=loss,
                    variance=variance,
                    duration_sec=duration_sec,
                    ok=ok,
                    error_msg=error_msg
                ))

        # Job mode not single-threaded; use either the shared ProcessPoolExecutor or ThreadPoolExecutor
        else:
            # Create a child process object or thread object for each GISJOIN validation job
            executors_list: list = []

            if self.request.worker_job_mode == JobMode.MULTITHREADED:
                with ThreadPoolExecutor(max_workers=10) as executor:
                    for spatial_allocation in self.request.allocations:
                        gis_join: str = spatial_allocation.gis_join
                        gis_join_count: int = self.gis_join_counts[gis_join]
                        info(f"Launching validation job for GISJOIN {spatial_allocation.gis_join}")
                        executors_list.append(
                            executor.submit(
                                validate_model,
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
                            validate_model,
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
                gis_join, allocation, loss, variance, ok, error_msg, duration_sec = \
                    future.result()
                metrics.append(ValidationMetric(
                    gis_join=gis_join,
                    allocation=allocation,
                    loss=loss,
                    variance=variance,
                    duration_sec=duration_sec,
                    ok=ok,
                    error_msg=error_msg
                ))

        info(f"metrics: {len(metrics)} responses")
        return metrics


# Independent function designed to be launched either within the same thread as the main process,
# on separate threads (but same Global Interpreter Lock) as the main process, or as separate child processes each
# with their own Global Interpreter Lock. Thus, all parameters have to be serializable.
def validate_model(
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

    import tensorflow as tf
    import pandas as pd
    import json
    import numpy as np
    import logging
    from logging import info, error
    from welford import Welford
    from sklearn.preprocessing import MinMaxScaler

    from overlay.db.querier import Querier

    ok = True
    error_msg = ""

    profiler: Timer = Timer()
    profiler.start()

    # Load Tensorflow model from disk (OS should cache in memory for future loads)
    model: tf.keras.Model = tf.keras.models.load_model(model_path)

    # Create or load persisted model metrics
    model_path_parts = model_path.split("/")[:-1]
    model_dir = "/".join(model_path_parts)
    model_metrics_path = f"{model_dir}/model_metrics_{gis_join}.json"
    if os.path.exists(model_metrics_path):
        info(f"P{model_metrics_path} exists, loading")
        with open(model_metrics_path, "r") as f:
            current_model_metrics = json.load(f)
    else:
        info(f"P{model_metrics_path} does not exist, initializing for first time")
        # First time calculating variance/errors for model
        current_model_metrics = {
            "gis_join": gis_join,
            "allocation": 0,
            "variance": 0.0,
            "m": 0.0,
            "s": 0.0,
            "loss": 0.0
        }

    if verbose:
        model.summary()

    info(f"mongo_host={mongo_host}, mongo_port={mongo_port}")
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

    allocation: int = len(features_df.index)
    info(f"Loaded Pandas DataFrame from MongoDB of size {allocation}")

    if allocation == 0:
        error_msg = f"No records found for GISJOIN {gis_join}"
        error(error_msg)
        return gis_join, 0, -1.0, -1.0, not ok, error_msg, 0.0

    # Normalize features, if requested
    if normalize_inputs:
        scaled = MinMaxScaler(feature_range=(0, 1)).fit_transform(features_df)
        features_df = pd.DataFrame(scaled, columns=features_df.columns)
        info(f"Normalized Pandas DataFrame")

    # Pop the label column off into its own DataFrame
    label_df = features_df.pop(label_field)

    if verbose:
        info(f"label_df: {label_df}")

    # Get predictions
    y_pred = model.predict(features_df, verbose=1 if verbose else 0)

    if verbose:
        info(f"y_pred: {y_pred}")

    # Use labels and predictions to evaluate the model
    y_true = np.array(label_df).reshape(-1, 1)

    if verbose:
        info(f"y_true: {y_true}")

    if loss_function == "MEAN_SQUARED_ERROR":
        info("MEAN_SQUARED_ERROR...")
        #loss = tf.reduce_mean(tf.square(tf.subtract(y_true, y_pred)))

        # Empirical Variance implementation for MSE
        # N_h = gis_join_count
        # e_k = np.square(y_true - y_pred)
        # sum_e_k = np.sum(e_k)
        # squared_sum_e_k = sum_e_k ** 2
        # squared_sum_e_k_div_by_N_h = squared_sum_e_k / N_h
        # e_k_squared = np.square(e_k)
        # sum_e_k_squared = np.sum(e_k_squared)
        # S_h_squared = (sum_e_k_squared - squared_sum_e_k_div_by_N_h) / (N_h - 1)
        # S_h = sqrt(S_h_squared)
        # variance: float = S_h

        # Numpy's built-in variance function for MSE
        squared_residuals = np.square(y_true - y_pred)
        m = np.mean(squared_residuals, axis=0)[0]
        loss = m
        s = (np.var(squared_residuals, axis=0, ddof=0) * squared_residuals.shape[0])[0]
        info(f"m = {m}, s = {s}, loss = {loss}")

    elif loss_function == "ROOT_MEAN_SQUARED_ERROR":
        info("ROOT_MEAN_SQUARED_ERROR...")
        #loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(y_true, y_pred))))
        squared_residuals = np.square(y_true - y_pred)
        m = np.mean(squared_residuals, axis=0)[0]
        loss = sqrt(m)
        s = (np.var(squared_residuals, axis=0, ddof=0) * squared_residuals.shape[0])[0]

    elif loss_function == "MEAN_ABSOLUTE_ERROR":
        info("MEAN_ABSOLUTE_ERROR...")
        absolute_residuals = np.abs(y_true - y_pred)
        loss = np.mean(absolute_residuals, axis=0)[0]
        m = np.mean(absolute_residuals, axis=0)[0]
        s = (np.var(absolute_residuals, axis=0, ddof=0) * absolute_residuals.shape[0])[0]

    else:
        profiler.stop()
        error_msg = f"Unsupported loss function {loss_function}"
        error(error_msg)
        return gis_join, allocation, -1.0, -1.0, not ok, error_msg, profiler.elapsed

    # Merging old metrics in with new metrics using Welford's method (if applicable)
    prev_allocation = current_model_metrics["allocation"]
    if prev_allocation > 0:
        prev_m = current_model_metrics["m"]
        prev_s = current_model_metrics["s"]
        new_allocation = prev_allocation + allocation
        delta = m - prev_m
        delta2 = delta * delta
        new_m = ((allocation * m) + (prev_allocation * prev_m)) / new_allocation
        new_s = s + prev_s + delta2 * (allocation * prev_allocation) / new_allocation
        new_loss = ((current_model_metrics["loss"] * prev_allocation) + (loss * allocation)) / new_allocation

        m = new_m
        s = new_s
        allocation = new_allocation
        loss = new_loss

    variance = s / allocation
    current_model_metrics["allocation"] = allocation
    current_model_metrics["loss"] = loss
    current_model_metrics["variance"] = variance
    current_model_metrics["m"] = m
    current_model_metrics["s"] = s
    with open(model_metrics_path, "w") as f:
        json.dump(current_model_metrics, f)

    profiler.stop()

    info(f"Evaluation results for GISJOIN {gis_join}: {loss}")
    return gis_join, allocation, loss, variance, ok, "", profiler.elapsed
