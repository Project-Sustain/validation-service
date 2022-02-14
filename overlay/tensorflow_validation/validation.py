import pymongo
import tensorflow as tf
import pandas as pd
from logging import info, error
from sklearn.preprocessing import MinMaxScaler

from overlay.validation_pb2 import ValidationMetric
from overlay.constants import DB_HOST, DB_PORT, DB_NAME
from overlay.db.querier import Querier


class TensorflowValidator:

    def __init__(self,
                 job_id: str,
                 models_dir: str,
                 model_type: str,
                 collection: str,
                 gis_join_key: str,
                 feature_fields: list,
                 label_field: str,
                 validation_metric: str,
                 normalize: bool):
        self.job_id = job_id
        self.models_dir = models_dir
        self.model_type = model_type
        self.collection = collection
        self.gis_join_key = gis_join_key
        self.feature_fields = feature_fields
        self.label_field = label_field
        self.validation_metric = validation_metric
        self.normalize = normalize
        self.querier = Querier(f"mongodb://{DB_HOST}:{DB_PORT}", DB_NAME)
        self.model = self.load_tf_model()

    def load_tf_model(self):
        # Load Tensorflow model from disk
        model_path = f"{self.models_dir}/{self.job_id}"
        info(f"Loading Tensorflow model from {model_path}")
        model = tf.keras.models.load_model(model_path)
        model.summary()
        return model

    def validate_gis_joins(self, gis_joins: list) -> list:
        metrics = []  # list of proto ValidationMetric objects
        for gis_join in gis_joins:
            info(f"Launching validation job for GISJOIN {gis_join}")
            loss = self.validate_gis_join(gis_join)

            if loss is None:
                metrics.append(ValidationMetric(
                    gis_join=gis_join,
                    loss=-1.0
                ))
            else:
                metrics.append(ValidationMetric(
                    gis_join=gis_join,
                    loss=loss
                ))

        self.querier.close()  # Close querier now that we are done using it
        return metrics

    def validate_gis_join(self, gis_join: str):
        # Query MongoDB for documents matching GISJOIN
        documents = self.querier.spatial_query(
            self.collection, self.gis_join_key, gis_join, self.feature_fields, self.label_field
        )

        # Load MongoDB Documents into Pandas DataFrame
        features_df = pd.DataFrame(list(documents))
        info(f"Loaded Pandas DataFrame from MongoDB, raw data:\n{features_df}")

        if len(features_df.index) == 0:
            error("DataFrame is empty! Returning None")
            return None

        # Normalize features, if requested
        if self.normalize:
            features_df = normalize_dataframe(features_df)
            info(f"Pandas DataFrame after normalization:\n{features_df}")

        # Pop the label column off into its own DataFrame
        label_df = features_df.pop(self.label_field)

        # Load model from disk
        validation_results = self.model.evaluate(features_df, label_df, batch_size=128, return_dict=True)
        info(f"Model validation results: {validation_results}")
        return validation_results['loss']


# Normalizes all the columns of a Pandas DataFrame using sklearn's Min-Max Feature Scaling.
def normalize_dataframe(dataframe):
    scaled = MinMaxScaler(feature_range=(0, 1)).fit_transform(dataframe)
    return pd.DataFrame(scaled, columns=dataframe.columns)
