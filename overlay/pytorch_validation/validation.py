import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from logging import info, error
from sklearn.preprocessing import MinMaxScaler
from concurrent.futures import ThreadPoolExecutor, as_completed

from overlay.validation_pb2 import ValidationMetric, MongoReadConfig, ValidationJobRequest
from overlay.db.querier import Querier
from overlay.constants import DB_HOST, DB_PORT, DB_NAME, MODELS_DIR
from overlay.tensorflow_validation.validation import normalize_dataframe


class PyTorchValidator:
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

    def load_pytorch_model(self, verbose=True):
        # Load PyTorch model from disk
        model_path = f"{MODELS_DIR}/{self.job_id}"

        # pick the first file. only one file (.pth file/pickled object) is supposed to be in the directory.
        file_name = os.listdir(model_path)[0]

        model_path += f'/{file_name}'

        info(f"Loading PyTorch model from {model_path}")

        model = torch.load(model_path)
        if verbose:
            model_description = f'{model}\nParameters:\n'
            for param in model.parameters():
                model_description = f'{param}\n'

            info(f"Model :{model_description}")

        return model

    def validate_gis_joins_synchronous(self, gis_joins: list) -> list:
        querier: Querier = Querier(
            mongo_host=self.mongo_host,
            mongo_port=self.mongo_port,
            db_name=self.database,
            read_preference=self.read_config.read_preference,
            read_concern=self.read_config.read_concern
        )
        model = self.load_pytorch_model()

        metrics = []  # list of ValidationMetric objects
        current = 1
        for gis_join in gis_joins:
            info(f"Launching validation job for GISJOIN {gis_join}, [{current}/{len(gis_joins)}]")
            score = self.validate_gis_join(gis_join, querier, model, False)
            metrics.append(ValidationMetric(
                gis_join=gis_join,
                loss=score
            ))
            current += 1

        querier.close()
        return metrics

    def validate_gis_joins_multithreaded(self, gis_joins: list) -> list:
        metrics = []

        # Iterate over all gis_joins and submit them for validation to the thread pool executor
        executors_list = []
        with ThreadPoolExecutor(max_workers=10) as executor:
            for gis_join in gis_joins:
                querier: Querier = Querier(mongo_host=self.mongo_host, mongo_port=self.mongo_port)
                model = self.load_pytorch_model()

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
        X = torch.from_numpy(features_df.astype(np.float32))
        y = torch.from_numpy(label_df.astype(np.float32))
        y = y.view(y.shape[0], 1)  # convert y to a column vector

        n_samples, n_features = X.shape
        info(f'n_samples: {n_samples}, n_features: {n_features}')

        # TODO: select criterion based on request
        criterion = nn.MSELoss()

        y_predicted = model(X)
        loss = criterion(y_predicted, y)

        info(f"Model validation results: {loss}")

        if is_concurrent:
            querier.close()

        return loss
