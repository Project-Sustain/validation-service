from logging import info, warning, error

from overlay.validation_pb2 import ValidationJobRequest
from overlay.validation import Validator
from overlay.profiler import Timer


class PyTorchValidator(Validator):

    def __init__(self, request: ValidationJobRequest, shared_executor, gis_join_counts):
        super().__init__(request, shared_executor, gis_join_counts)
        self.validate_model_function = validate_model


# Independent function designed to be launched either within the same thread as the main process,
# on separate threads (but same Global Interpreter Lock) as the main process, or as separate child processes each
# with their own Global Interpreter Lock. Thus, all parameters have to be serializable.
def validate_model(
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
        verbose: bool = True) -> (str, int, float, float, bool, str, float):
    # Returns the gis_join, allocation, loss, variance, ok status, error message, and duration

    import torch
    import os
    import json
    import pandas as pd
    import numpy as np
    import gc
    from sklearn.preprocessing import MinMaxScaler
    from math import sqrt

    from overlay.db.querier import Querier

    ok = True
    error_msg = ""

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
        info(f"P{model_metrics_path} exists, loading")
        with open(model_metrics_path, "r") as f:
            current_model_metrics = json.load(f)
    else:
        info(f"P{model_metrics_path} does not exist, initializing for first time")
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

        info(f"Model :{model_description}")

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
    info(f"Loaded Pandas DataFrame from MongoDB of size {len(features_df.index)}")

    if allocation == 0:
        error_msg = f"No records found for GISJOIN {gis_join}"
        error(error_msg)
        return gis_join, 0, -1.0, -1.0, not ok, error_msg, 0.0

    # Normalize features, if requested
    if normalize_inputs:
        scaled = MinMaxScaler(feature_range=(0, 1)).fit_transform(features_df)
        features_df = pd.DataFrame(scaled, columns=features_df.columns)
        info(f"Normalized Pandas DataFrame")

    # Pop the label column off into its own DataFrame
    label_df = features_df.pop(label_field)

    if verbose:
        info(f"label_df: {label_df}")

    # evaluate model
    inputs_numpy = features_df.values.astype(np.float32)
    y_true_numpy = label_df.values.astype(np.float32)
    inputs: torch.Tensor = torch.from_numpy(inputs_numpy)
    y_true: torch.Tensor = torch.from_numpy(y_true_numpy)
    y_true = y_true.view(y_true.shape[0], 1)  # convert y to a column vector

    n_samples, n_features = inputs.shape
    info(f'n_samples: {n_samples}, n_features: {n_features}')

    if loss_function == "MEAN_ABSOLUTE_ERROR":
        criterion = torch.nn.L1Loss()
        y_predicted = model(inputs)
        y_predicted_numpy = y_predicted.numpy()
        loss = criterion(y_predicted, y_true)
        absolute_residuals = np.absolute(y_predicted_numpy - y_true_numpy)

    elif loss_function == "MEAN_SQUARED_ERROR":
        # with torch.set_grad_enabled(False):
        #     linear.eval()
        #     print(linear.weight.requires_grad)
        with torch.no_grad():
            criterion = torch.nn.MSELoss()
            y_predicted = model(inputs)
            y_predicted_numpy = y_predicted.detach().numpy()
            loss = criterion(y_predicted, y_true)
            squared_residuals = np.square(y_predicted_numpy - y_true_numpy)

    elif loss_function == "ROOT_MEAN_SQUARED_ERROR":
        criterion = torch.nn.MSELoss()
        y_predicted = model(inputs)
        y_predicted_numpy = y_predicted.cpu().numpy()
        loss = sqrt(criterion(y_predicted, y_true))
        squared_residuals = np.square(y_predicted_numpy - y_true_numpy)

    elif loss_function == "NEGATIVE_LOG_LIKELIHOOD_LOSS":
        criterion = torch.nn.NLLLoss()
        y_predicted = model(inputs)
        y_predicted_numpy = y_predicted.cpu().numpy()
        loss = criterion(y_predicted, y_true)
        # TODO: Calculate NLL variances

    elif loss_function == "CROSS_ENTROPY_LOSS":
        criterion = torch.nn.CrossEntropyLoss()
        y_predicted = model(inputs)
        y_predicted_numpy = y_predicted.cpu().numpy()
        loss = criterion(y_predicted, y_true)
        # TODO: Calculate CE variances

    else:
        profiler.stop()
        error_msg = f"PyTorch validation: Unknown loss function: {loss_function}"
        warning(error_msg)
        return gis_join, allocation, -1.0, -1.0, not ok, error_msg, profiler.elapsed

    del model  # hopefully this frees up memory
    del inputs
    del y_true
    del y_predicted
    gc.collect()
    profiler.stop()
    variance_of_residuals = 0.0

    info(f"Evaluation results for GISJOIN {gis_join}: {loss}")
    return gis_join, allocation, loss, variance_of_residuals, ok, "", profiler.elapsed
