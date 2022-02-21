import pickle
from logging import info, error

import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler
from concurrent.futures import ThreadPoolExecutor, as_completed

from overlay.constants import DB_HOST, DB_PORT, DB_NAME
from overlay.db.querier import Querier
from overlay.tensorflow_validation.validation import normalize_dataframe
from overlay.validation_pb2 import ValidationMetric


class ScikitLearnValidator:
    def __init__(self,
                 job_id: str,
                 models_dir: str,
                 model_type: str,
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
        self.collection = collection
        self.gis_join_key = gis_join_key
        self.feature_fields = feature_fields
        self.label_field = label_field
        self.validation_metric = validation_metric
        self.normalize = normalize
        self.limit = limit
        self.sample_rate = sample_rate
        info(f"ScikitLearnValidator(): limit={self.limit}, sample_rate={self.sample_rate}")

    def load_sklearn_model(self, verbose=False):
        # Load ScikitLearn model from disk
        model_path = f"{self.models_dir}/{self.job_id}"
        info(f"Loading ScikitLearn model from {model_path}")
        model = pickle.load(open(model_path, 'rb'))
        # TODO: use verbose
        return model

    def validate_gis_joins_synchronous(self, gis_joins: list) -> list:
        querier: Querier = Querier(f"mongodb://{DB_HOST}:{DB_PORT}", DB_NAME)
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

    def validate_gis_join(self, gis_join: str, querier: Querier, model, is_concurrent: bool) -> float:
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
        features_df: DataFrame = pd.DataFrame(list(documents))
        # TODO: use is_concurrent

        if len(features_df.index) == 0:
            error("DataFrame is empty! Returning -1.0 for loss")
            return -1.0

        # Normalize features, if requested
        if self.normalize:
            features_df = normalize_dataframe(features_df)

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
