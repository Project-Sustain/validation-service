from loguru import logger

from overlay.profiler import Timer
from overlay.validation import Validator
from overlay.validation_pb2 import ValidationJobRequest, ModelCategory


class ScikitLearnValidator(Validator):

    def __init__(self, request: ValidationJobRequest, shared_executor, gis_join_counts):
        super().__init__(request, shared_executor, gis_join_counts)
        model_category_name = ModelCategory.Name(request.model_category)
        logger.debug(f"ScikitLearnValidator::__init__(): model_category: {model_category_name}")
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
    # Returns the gis_join, status, error_message, results

    import pandas as pd
    import os
    import json
    import pickle
    from sklearn.preprocessing import MinMaxScaler
    from sklearn import metrics

    from overlay.db.querier import Querier

    logger.info(f"Starting ScikitLearnValidator::validate_classification_model()")

    ok = True
    error_msg = ""
    iteration = 0
    profiler: Timer = Timer()
    profiler.start()

    # Load ScikitLearn classification model from disk
    logger.info(f"Loading Scikit-Learn classification model from {model_path}")
    model = pickle.load(open(model_path, 'rb'))

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
        model_type = type(model).__name__
        logger.info(f"Model type (from binary): {model_type}")

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
        logger.info(f"Normalized Pandas DataFrame")

    # Pop the label column off into its own DataFrame
    label_df = features_df.pop(label_field)

    if verbose:
        logger.info(f"label_df: {label_df}")

    # Get predictions (clasification)
    inputs_numpy = features_df.to_numpy()
    y_true = label_df.to_numpy()
    y_pred_class = model.predict(inputs_numpy)

    if verbose:
        logger.info(f"y_true: {y_true}")

    # calculate accuracy (percentage of correct predictions)
    accuracy = metrics.accuracy_score(y_true, y_pred_class)
    logger.success(f"Accuracy: {accuracy}")

    # value counts
    # TODO: check conversion format of y_true
    # logger.info(f"Value counts: {y_true.value_counts()}")

    # percentage of ones
    logger.success(f"Percentage of 1s: {y_true.mean()}")

    # percentage of zeroes
    logger.success(f"Percentage of 0s: {1 - y_true.mean()}")

    # null accuracy (for binary classification)
    logger.success(f"Null accuracy: {max(y_true.mean(), 1 - y_true.mean())}")

    # save confusion matrix and slice into four pieces
    confusion_matrix = metrics.confusion_matrix(y_true, y_pred_class)

    logger.success(f"Confusion matrix: {confusion_matrix}")

    TP = confusion_matrix[1, 1]
    TN = confusion_matrix[0, 0]
    FP = confusion_matrix[0, 1]
    FN = confusion_matrix[1, 0]

    logger.success(f"False positivity rate: {TN / (TN + FP)}")
    precision = metrics.precision_score(y_true, y_pred_class)
    recall = metrics.recall_score(y_true, y_pred_class)
    logger.success(f"Precision: {precision}")
    logger.success(f"Recall: {recall}")

    # ROC Curves and Area Under the Curve (AUC)
    y_pred_prob = model.predict_proba(features_df)[:, 1]
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred_prob)

    def evaluate_threshold(threshold):
        sensitivity = tpr[thresholds > threshold][-1]
        specificity = 1 - fpr[thresholds > threshold][-1]
        return [sensitivity, specificity]

    sensitivity1, specificity1 = evaluate_threshold(0.5)
    sensitivity2, specificity2 = evaluate_threshold(0.3)

    roc_auc_score = metrics.roc_auc_score(y_true, y_pred_prob)
    logger.success(f"roc_auc_score: {roc_auc_score}")

    # TODO: MVC design patterns
    # pivot the data for different views
    # implement a data structure
    # where the client pivots easily
    # partial streaming on the client side

    # raise NotImplementedError("validate_classification_model() is not implemented for class ScikitLearnValidator.")
    logger.debug(f"Returning GISJOIN: {gis_join}")
    return gis_join, 0, "no_error", "{sample_results}"
