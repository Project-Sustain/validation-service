from logging import info, error

from overlay.profiler import Timer
from overlay.validation_pb2 import ValidationJobRequest
from overlay.validation import Validator


class ScikitLearnValidator(Validator):

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
        verbose: bool = True) -> (str, int, float, float, int, bool, str, float):
    # Returns the gis_join, allocation, loss, variance, iteration, ok status, error message, and duration

    import pandas as pd
    import os
    import json
    import pickle
    import numpy as np
    from math import sqrt
    from sklearn.preprocessing import MinMaxScaler

    from overlay.db.querier import Querier

    ok = True
    error_msg = ""
    iteration = 0

    profiler: Timer = Timer()
    profiler.start()

    # Load ScikitLearn model from disk
    info(f"Loading Scikit-Learn model from {model_path}")
    model = pickle.load(open(model_path, 'rb'))

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
        model_type = type(model).__name__
        info(f"Model type (from binary): {model_type}")
        if model_type == "LinearRegression":
            info(f"Model Description: Coefficients: {model.coef_}, Intercept: {model.intercept_}")
        elif model_type == "GradientBoostingRegressor":
            info(f"Model Description(feature_importances: {model.feature_importances_},"
                 f"oob_improvement: {model.oob_improvement_},"
                 f"train_score: {model.train_score_},"
                 f"loss: {model.loss_},"
                 f"init_: {model.init_},"
                 f"estimators: {model.estimators_},"
                 f"n_classes: {model.n_classes_},"
                 f"n_estimators: {model.n_estimators_},"
                 f"max_features: {model.max_features_})")
        elif model_type == "SVR":
            info(f"Model Description(class_weight: {model.class_weight_},"
                 f"coef: {model.coef_},"
                 f"dual_coef: {model.dual_coef_},"
                 f"fit_status: {model.fit_status_},"
                 f"intercept: {model.intercept_},"
                 f"n_support: {model.n_support_},"
                 f"shape_fit: {model.shape_fit_},"
                 f"support: {model.support_},"
                 f"support_vectors_: {model.support_vectors_})")
        elif model_type == "RandomForestRegressor":
            info(f"Selecting RandomForestRegressor")
            # info(f"Model Description(base_estimator: {model.base_estimator_},"
            #      f"estimators: {model.estimators_}),"
            #      f"feature_importances_: {model.feature_importances_},"
            #      f"n_features_in: {model.n_features_in},"
            #      f"n_outputs: {model.n_outputs_},"
            #      f"oob_score: {model.oob_score_},"
            #      f"oob_prediction: {model.oob_score_}")
        else:
            error(f"Unsupported model type: {model_type}")

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

    # Get predictions
    inputs_numpy = features_df.to_numpy()
    y_true = label_df.to_numpy()
    y_pred = model.predict(inputs_numpy)

    if verbose:
        info(f"y_true: {y_true}")

    if loss_function == "MEAN_SQUARED_ERROR":
        info("MEAN_SQUARED_ERROR...")
        squared_residuals = np.square(y_true - y_pred)
        m = np.mean(squared_residuals, axis=0)
        loss = m
        s = (np.var(squared_residuals, axis=0, ddof=0) * squared_residuals.shape[0])
        info(f"m = {m}, s = {s}, loss = {loss}")

    elif loss_function == "ROOT_MEAN_SQUARED_ERROR":
        info("MEAN_SQUARED_ERROR...")
        squared_residuals = np.square(y_true - y_pred)
        m = np.mean(squared_residuals, axis=0)
        loss = sqrt(m)
        s = (np.var(squared_residuals, axis=0, ddof=0) * squared_residuals.shape[0])

    elif loss_function == "MEAN_ABSOLUTE_ERROR":
        info("MEAN_ABSOLUTE_ERROR...")
        absolute_residuals = np.abs(y_true - y_pred)
        loss = np.mean(absolute_residuals, axis=0)
        m = np.mean(absolute_residuals, axis=0)[0]
        s = (np.var(absolute_residuals, axis=0, ddof=0) * absolute_residuals.shape[0])

    else:
        profiler.stop()
        error_msg = f"Unsupported loss function {loss_function}"
        error(error_msg)
        return gis_join, allocation, -1.0, -1.0, iteration, not ok, error_msg, profiler.elapsed

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
