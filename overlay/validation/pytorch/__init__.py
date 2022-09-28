from logging import info, warning, error

from overlay.validation_pb2 import ValidationJobRequest, ModelCategory
from overlay.validation import Validator
from overlay.profiler import Timer


class PyTorchValidator(Validator):

    def __init__(self, request: ValidationJobRequest, shared_executor, gis_join_counts):
        super().__init__(request, shared_executor, gis_join_counts)


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
        verbose: bool = True) -> (str, int, float, float, bool, str, float):
    # Returns the gis_join, allocation, loss, variance, ok status, error message, and duration
    raise NotImplementedError("validate_classification_model() is not implemented for class PyTorchValidator.")


# Independent function designed to be launched either within the same thread as the main process,
# on separate threads (but same Global Interpreter Lock) as the main process, or as separate child processes each
# with their own Global Interpreter Lock. Thus, all parameters have to be serializable.
def validate_regression_model(
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
        verbose: bool = True) -> (str, int, float, float, int, bool, str, float):
    # Returns the gis_join, allocation, loss, variance, ok status, error message, and duration

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

        info(f"Model: {model_description}")
        info('Logging model.code...')
        info(model.code)

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
        return gis_join, 0, -1.0, -1.0, iteration, not ok, error_msg, 0.0

    # Normalize features, if requested
    if normalize_inputs:
        scaled = MinMaxScaler(feature_range=(0, 1)).fit_transform(features_df)
        features_df = pd.DataFrame(scaled, columns=features_df.columns)
        info(f"Normalized Pandas DataFrame")

    # Pop the label column off into its own DataFrame
    label_df = features_df.pop(label_field)

    if verbose:
        info(f"label_df: {label_df}")

    # Create Tensors from Pandas input/output dataframes
    inputs_tensor = torch.tensor(features_df.values, dtype=torch.float32, requires_grad=False)
    y_true_tensor = torch.tensor(label_df.values, dtype=torch.float32, requires_grad=False)
    y_true_tensor = y_true_tensor.view(y_true_tensor.shape[0], 1).squeeze(-1)

    n_samples, n_features = inputs_tensor.shape
    info(f'n_samples: {n_samples}, n_features: {n_features}')

    # Get model predictions
    with torch.no_grad():
        # criterion = torch.nn.MSELoss()
        y_predicted_tensor = model(inputs_tensor)
        info(f"y_predicted: {y_predicted_tensor}")
        info(f"y_true_tensor: {y_true_tensor}")
        y_predicted_numpy = y_predicted_tensor.numpy()
        y_true_numpy = y_true_tensor.numpy()

    if loss_function == "MEAN_ABSOLUTE_ERROR":
        info("MEAN_ABSOLUTE_ERROR...")
        absolute_residuals = np.abs(y_true_numpy - y_predicted_numpy)
        m = np.mean(absolute_residuals, axis=0).item()
        loss = m
        s = (np.var(absolute_residuals, axis=0, ddof=0) * absolute_residuals.shape[0]).item()

    elif loss_function == "MEAN_SQUARED_ERROR":
        info("MEAN_SQUARED_ERROR...")
        squared_residuals = np.square(y_true_numpy - y_predicted_numpy)
        info(f"squared_residuals: {squared_residuals}")
        m = np.mean(squared_residuals, axis=0).item()
        loss = m
        s = (np.var(squared_residuals, axis=0, ddof=0) * squared_residuals.shape[0]).item()

    elif loss_function == "ROOT_MEAN_SQUARED_ERROR":
        info("ROOT_MEAN_SQUARED_ERROR...")
        squared_residuals = np.square(y_true_numpy - y_predicted_numpy)
        m = np.mean(squared_residuals, axis=0).item()
        loss = sqrt(m)
        s = (np.var(squared_residuals, axis=0, ddof=0) * squared_residuals.shape[0]).item()

    elif loss_function == "NEGATIVE_LOG_LIKELIHOOD_LOSS":
        profiler.stop()
        error_msg = "PyTorch validation: negative log-likelihood loss unimplemented"
        warning(error_msg)
        return gis_join, allocation, -1.0, -1.0, iteration, not ok, error_msg, profiler.elapsed

    elif loss_function == "CROSS_ENTROPY_LOSS":
        profiler.stop()
        error_msg = "PyTorch validation: cross-entropy loss unimplemented"
        warning(error_msg)
        return gis_join, allocation, -1.0, -1.0, iteration, not ok, error_msg, profiler.elapsed

    else:
        profiler.stop()
        error_msg = f"PyTorch validation: Unknown loss function: {loss_function}"
        warning(error_msg)
        return gis_join, allocation, -1.0, -1.0, iteration, not ok, error_msg, profiler.elapsed

    info(f"m = {m}, s = {s}, loss = {loss}")

    # hopefully this frees up memory
    del model
    del inputs_tensor
    del y_true_tensor
    del y_predicted_tensor

    # Merging old metrics in with new metrics using Welford's method (if applicable)
    prev_allocation = current_model_metrics["allocation"]
    if prev_allocation > 0:
        iteration = 1
        prev_m = current_model_metrics["m"]
        prev_s = current_model_metrics["s"]
        new_allocation = prev_allocation + allocation
        delta = m - prev_m
        delta2 = delta * delta
        new_m = ((allocation * m) + (prev_allocation * prev_m)) / new_allocation
        new_s = s + prev_s + delta2 * (allocation * prev_allocation) / new_allocation
        new_loss = ((current_model_metrics["loss"] * prev_allocation) + (loss * allocation)) / new_allocation

        m = new_m
        s = new_s
        allocation = new_allocation
        loss = new_loss

    variance = s / allocation
    current_model_metrics["allocation"] = allocation
    current_model_metrics["loss"] = loss
    current_model_metrics["variance"] = variance
    current_model_metrics["m"] = m
    current_model_metrics["s"] = s
    with open(model_metrics_path, "w") as f:
        json.dump(current_model_metrics, f)

    profiler.stop()

    info(f"Evaluation results for GISJOIN {gis_join}: {loss}")
    return gis_join, allocation, loss, variance, iteration, ok, "", profiler.elapsed
