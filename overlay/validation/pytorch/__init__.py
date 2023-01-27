from loguru import logger

from overlay.validation_pb2 import ValidationJobRequest, ModelCategory
from overlay.validation import Validator
from overlay.profiler import Timer


class PyTorchValidator(Validator):

    def __init__(self, request: ValidationJobRequest, shared_executor, gis_join_counts):
        super().__init__(request, shared_executor, gis_join_counts)
        model_category_name = ModelCategory.Name(request.model_category)
        logger.debug(f"PyTorchValidator::__init__(): model_category: {model_category_name}")
        if str(model_category_name) == "REGRESSION":
            logger.error("Regression job submitted! The system supports classification only.")
        elif str(model_category_name) == "CLASSIFICATION":
            self.validate_model_function = validate_classification_model
        else:
            logger.error(f"Unsupported model category: {model_category_name}")


# Independent function designed to be launched either within the same thread as the main process,
# on separate threads (but same Global Interpreter Lock) as the main process, or as separate child processes each
# with their own Global Interpreter Lock. Thus, all parameters have to be serializable.
def validate_classification_model(
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
        verbose: bool = True) -> (str, int, str, str):
    # Returns the gis_join, status, error_msg, results

    import torch
    import os
    import json
    import pandas as pd
    import numpy as np
    import gc
    from sklearn.preprocessing import MinMaxScaler
    from math import sqrt
    from torch.utils.data import Dataset, DataLoader

    from overlay.db.querier import Querier

    ok = True
    error_msg = ""
    iteration = 0

    profiler: Timer = Timer()
    profiler.start()

    # Load TorchScript PyTorch model from disk (OS should cache in memory for future loads)
    model = torch.jit.load(model_path)
    model.eval()

    # Create or load persisted model metrics
    model_path_parts = model_path.split("/")[:-1]
    model_dir = "/".join(model_path_parts)
    model_metrics_path = f"{model_dir}/model_metrics_{gis_join}.json"
    if os.path.exists(model_metrics_path):
        logger.info(f"P{model_metrics_path} exists, loading")
        with open(model_metrics_path, "r") as f:
            current_model_metrics = json.load(f)
    else:
        logger.info(f"P{model_metrics_path} does not exist, initializing for first time")
        # First time calculating variance/errors for model
        current_model_metrics = {
            "gis_join": gis_join,
            "allocation": 0,
            "variance": 0.0,
            "m": 0.0,
            "s": 0.0,
            "loss": 0.0
        }

    if verbose:
        model_description = f'{model}\nParameters:\n'
        for param in model.parameters():
            model_description = f'{param}\n'

        logger.info(f"Model: {model_description}")
        logger.info('Logging model.code...')
        logger.info(model.code)

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
    logger.success(f"Loaded Pandas DataFrame from MongoDB of size {len(features_df.index)}")

    if allocation == 0:
        error_msg = f"No records found for GISJOIN {gis_join}"
        logger.error(error_msg)
        return gis_join, 0, -1.0, -1.0, iteration, not ok, error_msg, 0.0

    # Normalize features, if requested
    if normalize_inputs:
        scaled = MinMaxScaler(feature_range=(0, 1)).fit_transform(features_df)
        features_df = pd.DataFrame(scaled, columns=features_df.columns)
        logger.success(f"Normalized Pandas DataFrame")

    # Pop the label column off into its own DataFrame
    label_df = features_df.pop(label_field)

    if verbose:
        logger.info(f"label_df: {label_df}")

    # Create Tensors from Pandas input/output dataframes
    inputs_tensor = torch.tensor(features_df.values, dtype=torch.float32, requires_grad=False)
    y_true_tensor = torch.tensor(label_df.values, dtype=torch.float32, requires_grad=False)
    y_true_tensor = y_true_tensor.view(y_true_tensor.shape[0], 1).squeeze(-1)

    n_samples, n_features = inputs_tensor.shape
    logger.success(f'n_samples: {n_samples}, n_features: {n_features}')

    # Get model predictions
    with torch.no_grad():
        # criterion = torch.nn.MSELoss()
        logger.info("PyTorchValidator::validate_classification_model(): creating y_predicted_tensor")
        y_predicted_tensor = model(inputs_tensor)
        logger.success(f"y_predicted: {y_predicted_tensor}")
        logger.success(f"y_true_tensor: {y_true_tensor}")
        y_predicted_numpy = y_predicted_tensor.numpy()
        y_true_numpy = y_true_tensor.numpy()

    # raise NotImplementedError("validate_classification_model() is not implemented for class PyTorchValidator.")
    logger.debug(f"Returning GISJOIN: {gis_join}")
    return gis_join, 0, "no_error", "{sample_results}"
