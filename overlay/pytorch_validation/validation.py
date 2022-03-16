import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from logging import info, error
from concurrent.futures import ThreadPoolExecutor, as_completed

from overlay.validation_pb2 import ValidationMetric, ValidationJobRequest, BudgetType, StaticBudget
from overlay.db.querier import Querier
from overlay.constants import MODELS_DIR
from overlay.tensorflow_validation.validation import normalize_dataframe


class PyTorchValidator:
    def __init__(self, request: ValidationJobRequest):
        self.request = request

    def load_pytorch_model(self, verbose=True):
        # Load PyTorch model from disk
        model_path = f"{MODELS_DIR}/{self.request.id}/{self.request.id}.pth"
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
            mongo_host=self.request.mongo_host,
            mongo_port=self.request.mongo_port,
            db_name=self.request.database,
            read_preference=self.request.read_config.read_preference,
            read_concern=self.request.read_config.read_concern
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
                querier: Querier = Querier(
                    mongo_host=self.request.mongo_host,
                    mongo_port=self.request.mongo_port,
                    db_name=self.request.database,
                    read_preference=self.request.read_config.read_preference,
                    read_concern=self.request.read_config.read_concern
                )
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

        if self.request.validation_budget.budget_type == BudgetType.STATIC_BUDGET:

            budget: StaticBudget = self.request.validation_budget.static_budget
            limit: int = budget.strata_limit
            sample_rate: float = budget.sample_rate

            documents = querier.spatial_query(
                self.request.collection,
                gis_join,
                self.request.feature_fields,
                self.request.label_field,
                limit,
                sample_rate
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
            if self.request.normalize:
                features_df = normalize_dataframe(features_df)
                if is_concurrent:
                    info(f"Normalized Pandas DataFrame")
                else:
                    info(f"Pandas DataFrame after normalization:\n{features_df}")

            # Pop the label column off into its own DataFrame
            label_df = features_df.pop(self.request.label_field)

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

        elif self.request.validation_budget.budget_type == BudgetType.INCREMENTAL_VARIANCE_BUDGET:
            error("Incremental variance budgeting not yet implemented!")
            return 0.0

        else:
            error(f"Unrecognized budget type '{self.request.validation_budget.budget_type}'")
            return 0.0