import concurrent.futures
import pickle
import os
from logging import info, error

import pandas as pd
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from overlay.validation_pb2 import ValidationMetric, ValidationJobRequest, BudgetType, StaticBudget, JobMode
from overlay.constants import MODELS_DIR
from overlay.db.querier import Querier
from overlay.tensorflow_validation.validation import normalize_dataframe
from overlay.validation_pb2 import ValidationMetric, ValidationJobRequest, BudgetType, StaticBudget


class ScikitLearnValidator:

    def __init__(self, request: ValidationJobRequest):
        self.request: ValidationJobRequest = request
        self.model_path = self.get_model_path()

    def get_model_path(self):
        model_path = f"{MODELS_DIR}/{self.request.id}/"
        first_entry = os.listdir(model_path)[0]
        model_path += first_entry
        return model_path

    def validate_gis_joins(self, verbose: bool = True) -> list:

        # Convert protobuf "repeated" fields type to a python list
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
                loss: float = validate_model(
                    gis_join=gis_join,
                    model_path=self.model_path,
                    feature_fields=feature_fields,
                    label_field=self.request.label_field,
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
                    loss=loss
                ))

        # Job mode not single-threaded; either multi-thread or multi-processed
        else:

            # Choose executor type
            executor_type = ProcessPoolExecutor if self.request.worker_job_mode == JobMode.MULTIPROCESSING \
                else ThreadPoolExecutor

            executors_list: list = []
            with executor_type(max_workers=10) as executor:

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
                            self.request.mongo_host,
                            self.request.mongo_port,
                            self.request.read_config.read_preference,
                            self.request.read_config.read_concern,
                            self.request.database,
                            self.request.collection,
                            strata_limit,
                            sample_rate,
                            self.request.normalize_inputs,
                            None,
                            verbose
                        )
                    )

            # Wait on all tasks to finish -- Iterate over completed tasks, get their result, and log/append to responses
            for future in as_completed(executors_list):
                info(future)
                loss = future.result()

                metrics.append(ValidationMetric(
                    gis_join=gis_join,
                    loss=loss
                ))

        return metrics


def validate_model(
        gis_join: str,
        model_path: str,
        feature_fields: list,
        label_field: str,
        mongo_host: str,
        mongo_port: int,
        read_preference: str,
        read_concern: str,
        collection: str,
        database: str,
        limit: int,
        sample_rate: float,
        normalize_inputs: bool,
        verbose: bool = True) -> float:
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
    score = model.score(features_df, label_df)
    info(f"Model validation results: {score}")

    return score
