import pickle
import os
from logging import info, error

import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

from overlay.constants import MODELS_DIR
from overlay.db.querier import Querier
from overlay.tensorflow_validation.validation import normalize_dataframe
from overlay.validation_pb2 import ValidationMetric, ValidationJobRequest, BudgetType, StaticBudget


class ScikitLearnValidator:
    def __init__(self, request: ValidationJobRequest):
        self.request = request

    def load_sklearn_model(self, verbose=True):
        # Load ScikitLearn model from disk
        model_path = f"{MODELS_DIR}/{self.request.job_id}/{self.request.job_id}.pickle"
        info(f"Loading Scikit-Learn model from {model_path}")
        model = pickle.load(open(model_path, 'rb'))

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
                     f"n_features: {model.n_features_},"
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
            else:
                error(f"Unsupported model type: {model_type}")

        return model

    def validate_gis_joins_synchronous(self, gis_joins: list) -> list:
        querier: Querier = Querier(
            mongo_host=self.request.mongo_host,
            mongo_port=self.request.mongo_port,
            db_name=self.request.database,
            read_preference=self.request.read_config.read_preference,
            read_concern=self.request.read_config.read_concern
        )
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

    def validate_gis_joins_multithreaded(self, gis_joins: list) -> list:
        metrics = []  # list of proto ValidationMetric objects

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
                model = self.load_sklearn_model()

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
            if self.request.normalize_inputs:
                features_df = normalize_dataframe(features_df)
                if is_concurrent:
                    info(f"Normalized Pandas DataFrame")
                else:
                    info(f"Pandas DataFrame after normalization:\n{features_df}")

            # Pop the label column off into its own DataFrame
            label_df = features_df.pop(self.request.label_field)

            # evaluate model
            X_test = features_df
            y_test = label_df

            score = model.score(X_test, y_test)
            info(f"Model validation results: {score}")

            if is_concurrent:
                querier.close()

            return score

        elif self.request.validation_budget.budget_type == BudgetType.INCREMENTAL_VARIANCE_BUDGET:
            error("Incremental variance budgeting not yet implemented!")
            return 0.0
        else:
            error(f"Unrecognized budget type '{self.request.validation_budget.budget_type}'")
            return 0.0
