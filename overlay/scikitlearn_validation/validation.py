import concurrent.futures
import pickle
import os
from logging import info, error

import pandas as pd
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

from sklearn.preprocessing import MinMaxScaler

from overlay.validation_pb2 import ValidationMetric, ValidationJobRequest, BudgetType, StaticBudget, JobMode, \
    LossFunction
from overlay.constants import MODELS_DIR
from overlay.db.querier import Querier
from overlay.validation_pb2 import ValidationMetric, ValidationJobRequest, BudgetType, StaticBudget
from overlay.profiler import Timer


class ScikitLearnValidator:

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

    import pandas as pd
    import pickle
    import numpy as np
    from welford import Welford
    from math import sqrt
    from sklearn.preprocessing import MinMaxScaler

    from overlay.db.querier import Querier

    ok = True
    error_msg = ""

    profiler: Timer = Timer()
    profiler.start()

    # Load ScikitLearn model from disk
    info(f"Loading Scikit-Learn model from {model_path}")
    model = pickle.load(open(model_path, 'rb'))

    if verbose:
        model_type = type(model).__name__
        info(f"Model type (from binary): {model_type}")
        if model_type == "LinearRegression":
            info(f"Model Description: Coefficients: {model.coef_}, Intercept: {model.intercept_}")
        elif model_type == "GradientBoostingRegressor":
            info(f"Model Description(feature_importances: {model.feature_importances_},"
                 f"oob_improvement: {model.oob_improvement_},"
                 f"train_score: {model.train_score_},"
                 f"loss: {model.loss_},"
                 f"init_: {model.init_},"
                 f"estimators: {model.estimators_},"
                 f"n_classes: {model.n_classes_},"
                 f"n_estimators: {model.n_estimators_},"
                 f"n_features: {model.n_features_},"
                 f"max_features: {model.max_features_})")
        elif model_type == "SVR":
            info(f"Model Description(class_weight: {model.class_weight_},"
                 f"coef: {model.coef_},"
                 f"dual_coef: {model.dual_coef_},"
                 f"fit_status: {model.fit_status_},"
                 f"intercept: {model.intercept_},"
                 f"n_support: {model.n_support_},"
                 f"shape_fit: {model.shape_fit_},"
                 f"support: {model.support_},"
                 f"support_vectors_: {model.support_vectors_})")
        else:
            error(f"Unsupported model type: {model_type}")

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
    info(f"Loaded Pandas DataFrame from MongoDB of size {len(features_df.index)}")

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
    inputs_numpy = features_df.values.astype(np.float32)
    y_true = label_df.values.astype(np.float32).reshape(-1, 1)
    y_pred = model.predict(inputs_numpy)

    if verbose:
        info(f"y_true: {y_true}")

    welford_variance_calculator = Welford()
    if loss_function == "MEAN_SQUARED_ERROR":
        info("MEAN_SQUARED_ERROR...")
        squared_residuals = np.square(y_true - y_pred)
        loss = np.mean(squared_residuals, axis=0)[0]
        welford_variance_calculator.add_all(squared_residuals)

    elif loss_function == "ROOT_MEAN_SQUARED_ERROR":
        info("MEAN_SQUARED_ERROR...")
        squared_residuals = np.square(y_true - y_pred)
        loss = sqrt(np.mean(squared_residuals, axis=0)[0])
        welford_variance_calculator.add_all(squared_residuals)

    elif loss_function == "MEAN_ABSOLUTE_ERROR":
        info("MEAN_ABSOLUTE_ERROR...")
        loss = np.mean(np.abs(y_true - y_pred), axis=0)[0]
        absolute_residuals = np.absolute(y_pred - y_true)
        welford_variance_calculator.add_all(absolute_residuals)

    else:
        profiler.stop()
        error_msg = f"Unsupported loss function {loss_function}"
        error(error_msg)
        return gis_join, allocation, -1.0, -1.0, not ok, error_msg, profiler.elapsed

    # evaluate model
    #score = model.score(features_df, label_df)
    profiler.stop()
    variance_of_residuals = welford_variance_calculator.var_p

    info(f"Evaluation results for GISJOIN {gis_join}: {loss}")
    return gis_join, allocation, loss, variance_of_residuals, ok, "", profiler.elapsed

