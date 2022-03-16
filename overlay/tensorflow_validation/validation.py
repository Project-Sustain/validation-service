import tensorflow as tf
import pandas as pd
from logging import info, error
from sklearn.preprocessing import MinMaxScaler
from concurrent.futures import ThreadPoolExecutor, as_completed

from overlay.validation_pb2 import ValidationMetric, MongoReadConfig, ValidationJobRequest, BudgetType, StaticBudget
from overlay.db.querier import Querier
from overlay.constants import DB_HOST, DB_PORT, DB_NAME, MODELS_DIR


class TensorflowValidator:

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
        self.validation_budget = request.validation_budget

    def load_tf_model(self, verbose=False):
        # Load Tensorflow model from disk
        model_path = f"{MODELS_DIR}/{self.job_id}"
        info(f"Loading Tensorflow model from {model_path}")
        model = tf.keras.models.load_model(model_path)
        if verbose:
            model.summary()
        return model

    def validate_gis_joins_synchronous(self, gis_joins: list) -> list:
        querier: Querier = Querier(
            mongo_host=self.mongo_host,
            mongo_port=self.mongo_port,
            db_name=self.database,
            read_preference=self.read_config.read_preference,
            read_concern=self.read_config.read_concern
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

        # Iterate over all gis_joins and submit them for validation to the thread pool executor
        executors_list = []
        with ThreadPoolExecutor(max_workers=10) as executor:
            for gis_join in gis_joins:
                querier: Querier = Querier(mongo_host=self.mongo_host, mongo_port=self.mongo_port)
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

        return metrics

    def validate_gis_join(self, gis_join: str, querier: Querier, model: tf.keras.Model, is_concurrent: bool) -> float:
        # Query MongoDB for documents matching GISJOIN

        if self.validation_budget.budget_type == BudgetType.STATIC_BUDGET:
            budget: StaticBudget = self.validation_budget.static_budget
            limit: int = budget.strata_limit
            sample_rate: float = budget.sample_rate

            documents = querier.spatial_query(
                self.collection,
                gis_join,
                self.feature_fields,
                self.label_field,
                limit,
                sample_rate
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

            # Evaluate model
            validation_results = model.evaluate(features_df, label_df, batch_size=128, return_dict=True, verbose=0)
            info(f"Model validation results: {validation_results}")

            # If the MongoDB driver connection is local to this thread/function, close it when done using it
            if is_concurrent:
                querier.close()

            return validation_results['loss']

        elif self.validation_budget.budget_type == BudgetType.INCREMENTAL_VARIANCE_BUDGET:
            error("Incremental variance budgeting not yet implemented!")
            return 0.0

        else:
            error(f"Unrecognized budget type '{self.validation_budget.budget_type}'")
            return 0.0


# Normalizes all the columns of a Pandas DataFrame using sklearn's Min-Max Feature Scaling.
def normalize_dataframe(dataframe):
    scaled = MinMaxScaler(feature_range=(0, 1)).fit_transform(dataframe)
    return pd.DataFrame(scaled, columns=dataframe.columns)
