import pandas as pd
import torch
from logging import info, error
from sklearn.preprocessing import MinMaxScaler
from concurrent.futures import ThreadPoolExecutor, as_completed

from overlay.validation_pb2 import ValidationMetric, MongoReadConfig, ValidationJobRequest
from overlay.db.querier import Querier
from overlay.constants import DB_HOST, DB_PORT, DB_NAME, MODELS_DIR


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
        info(f"Loading PyTorch model from {model_path}")

        model = torch.load(model_path)
        if verbose:
            model_description = f'{model}\nParameters:\n'
            for param in model.parameters():
                model_description = f'{param}\n'

            info(f"Model :{model_description}")

        return model

    def validate_gis_joins_synchronously(self, gis_joins: list) -> list:
        pass
