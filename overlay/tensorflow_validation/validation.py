import tensorflow as tf
import pandas as pd
from logging import info, error
from sklearn.preprocessing import MinMaxScaler
from concurrent.futures import ThreadPoolExecutor, as_completed

from overlay.validation_pb2 import ValidationMetric, MongoReadConfig
from overlay.db.querier import Querier


class TensorflowValidator:

    def __init__(self,
                 job_id: str,
                 models_dir: str,
                 model_type: str,
                 mongo_host: str,
                 mongo_port: int,
                 read_config: MongoReadConfig,
                 collection: str,
                 gis_join_key: str,
                 feature_fields: list,
                 label_field: str,
                 validation_metric: str,
                 normalize: bool,
                 limit: int,
                 sample_rate: float):

        self.job_id = job_id
        self.models_dir = models_dir
        self.model_type = model_type
        self.mongo_host = mongo_host
        self.mongo_port = mongo_port
        self.read_config = read_config
        self.collection = collection
        self.gis_join_key = gis_join_key
        self.feature_fields = feature_fields
        self.label_field = label_field
        self.validation_metric = validation_metric
        self.normalize = normalize
        self.limit = limit
        self.sample_rate = sample_rate

    def load_tf_model(self, verbose=False):
        # Load Tensorflow model from disk
        model_path = f"{self.models_dir}/{self.job_id}"
        info(f"Loading Tensorflow model from {model_path}")
        model = tf.keras.models.load_model(model_path)
        if verbose:
            model.summary()
        return model

    def validate_gis_joins_synchronous(self, gis_joins: list) -> list:
        querier: Querier = Querier(mongo_host=self.mongo_host, mongo_port=self.mongo_port)
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
        info(f"Using limit={self.limit}, and sample_rate={self.sample_rate}")

        documents = querier.spatial_query(
            self.collection,
            self.gis_join_key,
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

        # Evaluate model
        validation_results = model.evaluate(features_df, label_df, batch_size=128, return_dict=True, verbose=0)
        info(f"Model validation results: {validation_results}")

        if is_concurrent:
            querier.close()

        return validation_results['loss']


# Normalizes all the columns of a Pandas DataFrame using sklearn's Min-Max Feature Scaling.
def normalize_dataframe(dataframe):
    scaled = MinMaxScaler(feature_range=(0, 1)).fit_transform(dataframe)
    return pd.DataFrame(scaled, columns=dataframe.columns)
