import concurrent.futures
import os

import tensorflow as tf
import pandas as pd
from logging import info, error
from sklearn.preprocessing import MinMaxScaler
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

from overlay.validation_pb2 import ValidationMetric, ValidationJobRequest, BudgetType, StaticBudget
from overlay.db.querier import Querier
from overlay.constants import MODELS_DIR


class TensorflowValidator:

    def __init__(self, request: ValidationJobRequest):
        self.request: ValidationJobRequest = request
        self.model_path = self.get_model_path()

    def load_tf_model(self, verbose=False):
        # Load Tensorflow model from disk
        model_path = f"{MODELS_DIR}/{self.request.id}/"
        first_entry = os.listdir(model_path)[0]
        model_path += first_entry
        info(f"Loading Tensorflow model from {model_path}")
        model = tf.keras.models.load_model(model_path)

        if verbose:
            model.summary()
        return model

    def get_model_path(self):
        model_path = f"{MODELS_DIR}/{self.request.id}/"
        first_entry = os.listdir(model_path)[0]
        model_path += first_entry
        return model_path

    def validate_gis_joins_synchronous(self, gis_joins: list) -> list:
        querier: Querier = Querier(
            mongo_host=self.request.mongo_host,
            mongo_port=self.request.mongo_port,
            db_name=self.request.database,
            read_preference=self.request.read_config.read_preference,
            read_concern=self.request.read_config.read_concern
        )
        model: tf.keras.Model = self.load_tf_model()

        metrics = []  # list of proto ValidationMetric objects
        current = 1
        for gis_join in gis_joins:
            info(f"Launching validation job for GISJOIN {gis_join}, [{current}/{len(gis_joins)}]")
            loss = self.validate_gis_join(gis_join, querier, model, False)
            metrics.append(ValidationMetric(
                gis_join=gis_join,
                loss=loss
            ))
            current += 1

        querier.close()  # Close querier now that we are done using it
        return metrics

    def validate_gis_joins_multithreaded(self, gis_joins: list) -> list:
        metrics = []  # list of proto ValidationMetric objects
        querier: Querier = Querier(
            mongo_host=self.request.mongo_host,
            mongo_port=self.request.mongo_port,
            db_name=self.request.database,
            read_preference=self.request.read_config.read_preference,
            read_concern=self.request.read_config.read_concern
        )

        # Iterate over all gis_joins and submit them for validation to the thread pool executor
        executors_list = []
        with ThreadPoolExecutor(max_workers=10) as executor:
            for gis_join in gis_joins:
                model: tf.keras.Model = self.load_tf_model()
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

        querier.close()

        return metrics

    # private helper method
    # def _multiprocess_helper(self, gis_join):
        # info(f"Launching validation job for GISJOIN {gis_join}, [concurrent/{len(gis_joins)}]")
        # info(f"Launching validation job for GISJOIN {gis_join}")
        # return test_func(gis_join, self.request.id,
        #                  self.request.feature_fields, self.request.label_field)

    def validate_gis_joins_multiprocessing(self, gis_joins: list) -> list:
        metrics = []  # list of proto ValidationMetric objects

        # Iterate over all gis_joins and submit them for validation to the thread pool executor
        executors_list = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
            # results = executor.map(self._multiprocess_helper, gis_joins)
            for gis_join in gis_joins:
                info(f"Launching validation job for GISJOIN {gis_join}, [concurrent/{len(gis_joins)}]")

                executors_list.append(
                    executor.submit(
                        validate_model,
                        gis_join,
                        self.model_path,
                        [feature_field for feature_field in self.request.feature_fields.items()],
                        self.request.label_field,
                        self.request.mongo_host,
                        self.request.mongo_port,
                        self.request.read_config.read_preference,
                        self.request.read_config.read_concern,
                        self.request.database,
                        self.request.collection,
                        self.request.validation_budget.static_budget.strata_limit,
                        self.request.validation_budget.static_budget.sample_rate
                    )
                )


        # Wait on all tasks to finish -- Iterate over completed tasks, get their result, and log/append to responses
        # for loss, gis_join in zip(results, gis_joins):
        #     metrics.append(ValidationMetric(
        #         gis_join=gis_join,
        #         loss=loss
        #     ))
        for future in as_completed(executors_list):
            info(future)
            loss = future.result()

            metrics.append(ValidationMetric(
                gis_join=gis_join,
                loss=loss
            ))

        return metrics

    def validate_gis_join(self, gis_join: str, querier: Querier, model: tf.keras.Model, is_concurrent: bool) -> float:
        # Query MongoDB for documents matching GISJOIN
        info(f"Launching validation task for GISJOIN {gis_join}")
        if self.request.validation_budget.budget_type == BudgetType.STATIC_BUDGET:
            budget: StaticBudget = self.request.validation_budget.static_budget
            limit: int = budget.strata_limit
            sample_rate: float = budget.sample_rate

            documents = querier.spatial_query(
                self.request.collection,
                gis_join,
                self.request.feature_fields,
                self.request.label_field,
                limit,
                sample_rate
            )

            # Load MongoDB Documents into Pandas DataFrame
            features_df = pd.DataFrame(list(documents))

            # If the MongoDB driver connection is local to this thread/function, close it when done using it
            # if is_concurrent:
            #     querier.close()

            if is_concurrent:
                info(f"Loaded Pandas DataFrame from MongoDB of size {len(features_df.index)}")
            else:
                info(f"Loaded Pandas DataFrame from MongoDB, raw data:\n{features_df}")

            if len(features_df.index) == 0:
                error("DataFrame is empty! Returning -1.0 for loss")
                return -1.0

            # Normalize features, if requested
            if self.request.normalize_inputs:
                features_df = normalize_dataframe(features_df)
                if is_concurrent:
                    info(f"Normalized Pandas DataFrame")
                else:
                    info(f"Pandas DataFrame after normalization:\n{features_df}")

            # Pop the label column off into its own DataFrame
            label_df = features_df.pop(self.request.label_field)

            # Evaluate model
            validation_results = model.evaluate(features_df, label_df, batch_size=128, return_dict=True, verbose=0)
            info(f"Model validation results: {validation_results}")

            return validation_results['loss']

        elif self.request.validation_budget.budget_type == BudgetType.INCREMENTAL_VARIANCE_BUDGET:
            error("Incremental variance budgeting not yet implemented!")
            return 0.0

        else:
            error(f"Unrecognized budget type '{self.request.validation_budget.budget_type}'")
            return 0.0


# Normalizes all the columns of a Pandas DataFrame using sklearn's Min-Max Feature Scaling.
def normalize_dataframe(dataframe):
    scaled = MinMaxScaler(feature_range=(0, 1)).fit_transform(dataframe)
    return pd.DataFrame(scaled, columns=dataframe.columns)


def validate_model(
        gis_join: str,
        model_path: str,
        feature_fields: list,
        label_field: str,
        mongo_host: str,
        mongo_port: int,
        read_preference: str,
        read_concern: str,
        database: str,
        collection: str,
        limit: int,
        sample_rate: float) -> float:

    # Load Tensorflow model from disk
    model: tf.keras.Model = tf.keras.models.load_model(model_path)
    model.summary()

    querier: Querier = Querier(
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

    # If the MongoDB driver connection is local to this thread/function, close it when done using it
    querier.close()

    info(f"Loaded Pandas DataFrame from MongoDB of size {len(features_df.index)}")

    if len(features_df.index) == 0:
        error("DataFrame is empty! Returning -1.0 for loss")
        return -1.0

    # Normalize features, if requested
    features_df = normalize_dataframe(features_df)
    info(f"Normalized Pandas DataFrame")

    # Pop the label column off into its own DataFrame
    label_df = features_df.pop(label_field)

    # Evaluate model
    validation_results = model.evaluate(features_df, label_df, batch_size=128, return_dict=True, verbose=0)
    info(f"Model validation results: {validation_results}")

    return validation_results['loss']
