from loguru import logger

import os
from math import sqrt

from overlay.validation_pb2 import ValidationJobRequest, ModelCategory
from overlay.profiler import Timer
from overlay.validation import Validator


class TensorflowValidator(Validator):

    def __init__(self, request: ValidationJobRequest, shared_executor, gis_join_counts):
        super().__init__(request, shared_executor, gis_join_counts)
        model_category_name = ModelCategory.Name(request.model_category)
        logger.info(f"TensorflowValidator::__init__(): model_category: {model_category_name}")
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

    import tensorflow as tf
    import pandas as pd
    import json
    import numpy as np
    import logging
    from logging import info, error
    from sklearn.preprocessing import MinMaxScaler

    from overlay.db.querier import Querier

    logger.info(f"Starting TensorflowValidator::validate_classification_model()")

    # raise NotImplementedError("validate_classification_model() is not implemented for class TensorflowValidator.")
    logger.debug(f"Returning GISJOIN: {gis_join}")
    return gis_join, 0, "no_error", "{sample_results}"
