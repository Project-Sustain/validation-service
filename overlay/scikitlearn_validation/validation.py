import pickle
import os
from logging import info, error

import pandas as pd
from pandas import DataFrame
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from concurrent.futures import ThreadPoolExecutor, as_completed

from overlay.constants import DB_HOST, DB_PORT, DB_NAME, MODELS_DIR
from overlay.db.querier import Querier
from overlay.tensorflow_validation.validation import normalize_dataframe
from overlay.validation_pb2 import ValidationMetric, ValidationJobRequest


class ScikitLearnValidator:
    def __init__(self, request: ValidationJobRequest):
        self.job_id = request.id
        self.model_type = request.model_category
        self.mongo_host = request.mongo_host
        self.mongo_port = request.mongo_port
        self.read_config = request.read_config
        self.database = request.database
        self.collection = request.collection
        self.feature_fields = request.feature_fields
        self.label_field = request.label_field
        self.validation_metric = request.validation_metric
        self.normalize = request.normalize_inputs
        self.limit = request.limit
        self.sample_rate = request.sample_rate
        info(f"ScikitLearnValidator(): limit={self.limit}, sample_rate={self.sample_rate}")

    def load_sklearn_model(self, verbose=True):
        # Load ScikitLearn model from disk
        model_path = f"{MODELS_DIR}/{self.job_id}"

        # pick the first file. only one file (pickled object) is supposed to be in the directory.
        file_name = os.listdir(model_path)[0]

        model_path += f'/{file_name}'

        model = pickle.load(open(model_path, 'rb'))

        info(f"Loading ScikitLearn model from {model_path}")

        if verbose:
            model_type = type(model).__name__
            info(f"Model type (from binary): {model_type}")
            if model_type == "LinearRegression":
                info(f"Model Description: Coefficients: {model.coef_}, Intercept: {model.intercept_}")
            elif model_type == "GradientBoostingRegressor":
                info(f"Model Description(feature_importances: {model.feature_importances_},"
                     f"oob_improvement: {model.oob_improvement_},"
                     f"train_score: {model.train_score_},"
                     f"loss: {model.losee_},"
                     f"init_: {model.init_},"
                     f"estimators: {model.estimators_},"
                     f"n_classes: {model.n_clases_},"
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

        return model

    def validate_gis_joins_synchronous(self, gis_joins: list) -> list:
        querier: Querier = Querier(
            mongo_host=self.mongo_host,
            mongo_port=self.mongo_port,
            db_name=self.database,
            read_preference=self.read_config.read_preference,
            read_concern=self.read_config.read_concern
        )
        model = self.load_sklearn_model()
        metrics = []  # list of ValidationMetric objects
        current = 1
        for gis_join in gis_joins:
            info(f"Launching validation job for GISJOIN {gis_join}, [{current}/{len(gis_joins)}]")
            # TODO: retrieve loss instead of accuracy
            score = self.validate_gis_join(gis_join, querier, model, False)
            metrics.append(ValidationMetric(
                gis_join=gis_join,
                loss=score
            ))
            current += 1

        querier.close()
        return metrics

    def validate_gis_joins_multithreaded(self, gis_joins: list) -> list:
        metrics = []  # list of proto ValidationMetric objects

        # Iterate over all gis_joins and submit them for validation to the thread pool executor
        executors_list = []
        with ThreadPoolExecutor(max_workers=10) as executor:
            for gis_join in gis_joins:
                querier: Querier = Querier(mongo_host=self.mongo_host, mongo_port=self.mongo_port)
                model = self.load_sklearn_model()

                info(f"Launching validation job for GISJOIN {gis_join}, [concurrent/{len(gis_joins)}]")
                executors_list.append(executor.submit(self.validate_gis_join, gis_join, querier, model, True))

        # Wait on all tasks to finish -- Iterate over completed tasks, get their result, and log/append to responses
        for future in as_completed(executors_list):
            info(future)
            loss = future.result()

            metrics.append(ValidationMetric(
                gis_join=gis_join,
                loss=loss
            ))

        return metrics

    def validate_gis_join(self, gis_join: str, querier: Querier, model, is_concurrent: bool) -> float:
        info(f"Using limit={self.limit}, and sample_rate={self.sample_rate}")

        documents = querier.spatial_query(
            self.collection,
            gis_join,
            self.feature_fields,
            self.label_field,
            self.limit,
            self.sample_rate
        )

        # Load MongoDB Documents into Pandas DataFrame
        features_df = pd.DataFrame(list(documents))
        if is_concurrent:
            info(f"Loaded Pandas DataFrame from MongoDB of size {len(features_df.index)}")
        else:
            info(f"Loaded Pandas DataFrame from MongoDB, raw data:\n{features_df}")

        if len(features_df.index) == 0:
            error("DataFrame is empty! Returning -1.0 for loss")
            return -1.0

        # Normalize features, if requested
        if self.normalize:
            features_df = normalize_dataframe(features_df)
            if is_concurrent:
                info(f"Normalized Pandas DataFrame")
            else:
                info(f"Pandas DataFrame after normalization:\n{features_df}")

        # Pop the label column off into its own DataFrame
        label_df = features_df.pop(self.label_field)

        # evaluate model
        X_test = features_df
        y_test = label_df

        score = model.score(X_test, y_test)
        info(f"Model validation results: {score}")

        if is_concurrent:
            querier.close()

        return score
