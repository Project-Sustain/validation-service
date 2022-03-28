import concurrent.futures
import os

import tensorflow as tf
import pandas as pd
import numpy as np
from logging import info, error
from sklearn.preprocessing import MinMaxScaler
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

from overlay.validation_pb2 import ValidationMetric, ValidationJobRequest, BudgetType, StaticBudget, JobMode, \
    LossFunction, ModelFramework, ModelCategory, MongoReadConfig, ValidationBudget
from overlay.db.querier import Querier
from overlay.profiler import Timer
from overlay.constants import MODELS_DIR


class TensorflowValidator:

    def __init__(self, request: ValidationJobRequest):
        self.request: ValidationJobRequest = request
        self.model_path = self.get_model_path()

    def get_model_path(self):
        model_path = f"{MODELS_DIR}/{self.request.id}/"
        first_entry = os.listdir(model_path)[0]
        model_path += first_entry
        return model_path

    def validate_gis_joins(self, verbose: bool = True) -> list:

        info(f"Launching validation job for {len(self.request.gis_joins)} GISJOINs")

        # Convert protobuf "repeated" field type to a python list
        feature_fields = []
        for feature_field in self.request.feature_fields:
            feature_fields.append(feature_field)

        # Capture validation budget parameters
        if self.request.validation_budget.budget_type == BudgetType.STATIC_BUDGET:
            budget: StaticBudget = self.request.validation_budget.static_budget
            strata_limit: int = budget.strata_limit
            sample_rate: float = budget.sample_rate
        elif self.request.validation_budget.budget_type == BudgetType.INCREMENTAL_VARIANCE_BUDGET:
            error("Incremental variance budgeting not yet implemented!")
            return []
        else:
            error(f"Unrecognized budget type '{self.request.validation_budget.budget_type}'")
            return []

        metrics = []  # list of protobuf ValidationMetric objects to return

        # Select job mode
        if self.request.worker_job_mode == JobMode.SYNCHRONOUS:

            # Make requests serially
            for gis_join in self.request.gis_joins:
                returned_gis_join, loss, ok, error_msg, duration_sec = validate_model(
                    gis_join=gis_join,
                    model_path=self.model_path,
                    feature_fields=feature_fields,
                    label_field=self.request.label_field,
                    loss_function=LossFunction.Name(self.request.loss_function),
                    mongo_host=self.request.mongo_host,
                    mongo_port=self.request.mongo_port,
                    read_preference=self.request.read_config.read_preference,
                    read_concern=self.request.read_config.read_concern,
                    database=self.request.database,
                    collection=self.request.collection,
                    limit=strata_limit,
                    sample_rate=sample_rate,
                    normalize_inputs=self.request.normalize_inputs,
                    verbose=verbose
                )

                metrics.append(ValidationMetric(
                    gis_join=returned_gis_join,
                    loss=loss,
                    duration_sec=duration_sec,
                    ok=ok,
                    error_msg=error_msg
                ))

        # Job mode not single-threaded; either multi-thread or multi-processed
        else:
            # Select either process or thread pool executor type
            executor_type = ProcessPoolExecutor if self.request.worker_job_mode == JobMode.MULTIPROCESSING \
                or self.request.worker_job_mode == JobMode.DEFAULT_JOB_MODE else ThreadPoolExecutor

            executors_list: list = []
            with executor_type(max_workers=8) as executor:

                # Create either a thread or child process object for each GISJOIN validation job
                for gis_join in self.request.gis_joins:

                    # info(f"Launching validation job for GISJOIN {gis_join}")
                    executors_list.append(
                        executor.submit(
                            validate_model,
                            gis_join,
                            self.model_path,
                            feature_fields,
                            self.request.label_field,
                            LossFunction.Name(self.request.loss_function),
                            self.request.mongo_host,
                            self.request.mongo_port,
                            self.request.read_config.read_preference,
                            self.request.read_config.read_concern,
                            self.request.database,
                            self.request.collection,
                            strata_limit,
                            sample_rate,
                            self.request.normalize_inputs,
                            False  # don't log summaries on concurrent model
                        )
                    )

            # Wait on all tasks to finish -- Iterate over completed tasks, get their result, and log/append to responses
            for future in as_completed(executors_list):
                gis_join, loss, ok, error_msg, duration_sec = future.result()
                metrics.append(ValidationMetric(
                    gis_join=gis_join,
                    loss=loss,
                    duration_sec=duration_sec,
                    ok=ok,
                    error_msg=error_msg
                ))

        info(f"metrics: {len(metrics)} responses")
        return metrics


# Independent function designed to be launched either within the same thread as the main process,
# on separate threads (but same Global Interpreter Lock) as the main process, or as separate child processes each
# with their own Global Interpreter Lock. Thus, all parameters have to be serializable.
def validate_model(
        gis_join: str,
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
        verbose: bool = True) -> (str, float, bool, str, float):  # Returns the gis_join, loss, ok status, error message, and duration

    profiler: Timer = Timer()
    profiler.start()

    # Load Tensorflow model from disk (OS should cache in memory for future loads)
    model: tf.keras.Model = tf.keras.models.load_model(model_path)

    if verbose:
        model.summary()

    info(f"mongo_host={mongo_host}, mongo_port={mongo_port}")
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

    info(f"Loaded Pandas DataFrame from MongoDB of size {len(features_df.index)}")

    if len(features_df.index) == 0:
        error_msg = f"No records found for GISJOIN {gis_join}"
        error(error_msg)
        return gis_join, -1.0, False, error_msg, 0.0

    # Normalize features, if requested
    if normalize_inputs:
        features_df = normalize_dataframe(features_df)
        info(f"Normalized Pandas DataFrame")

    # Pop the label column off into its own DataFrame
    label_df = features_df.pop(label_field)

    if verbose:
        info(f"label_df: {label_df}")

    # Get predictions
    y_pred = model.predict(features_df, verbose=1 if verbose else 0)

    if verbose:
        info(f"y_pred: {y_pred}")

    # Use labels and predictions to evaluate the model
    y_true = np.array(label_df).reshape(-1, 1)

    if verbose:
        info(f"y_true: {y_true}")

    loss: float = 0.0
    if loss_function == "MEAN_SQUARED_ERROR":
        info("MEAN_SQUARED_ERROR...")
        loss = tf.reduce_mean(tf.square(tf.subtract(y_true, y_pred)))
    elif loss_function == "ROOT_MEAN_SQUARED_ERROR":
        info("ROOT_MEAN_SQUARED_ERROR...")
        loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(y_true, y_pred))))
    elif loss_function == "MEAN_ABSOLUTE_ERROR":
        info("MEAN_ABSOLUTE_ERROR...")
        loss = np.mean(np.abs(y_true - y_pred), axis=0)[0]
    else:
        profiler.stop()
        error_msg = f"Unsupported loss function {loss_function}"
        error(error_msg)
        return gis_join, -1.0, False, error_msg, profiler.elapsed

    profiler.stop()
    info(f"Evaluation results for GISJOIN {gis_join}: {loss}")
    return gis_join, loss, True, "", profiler.elapsed


# Normalizes all the columns of a Pandas DataFrame using Scikit-Learn Min-Max Feature Scaling.
def normalize_dataframe(dataframe):
    scaled = MinMaxScaler(feature_range=(0, 1)).fit_transform(dataframe)
    return pd.DataFrame(scaled, columns=dataframe.columns)


def test():
    validator: TensorflowValidator = TensorflowValidator(
        request=ValidationJobRequest(
            id="test",
            master_job_mode=JobMode.ASYNCHRONOUS,
            worker_job_mode=JobMode.MULTIPROCESSING,
            model_framework=ModelFramework.TENSORFLOW,
            model_category=ModelCategory.REGRESSION,
            mongo_host="lattice-150",
            mongo_port=27018,
            read_config=MongoReadConfig(
                read_preference="primary",
                read_concern="available"
            ),
            database="sustaindb",
            collection="noaa_nam",
            feature_fields=[
                "PRESSURE_AT_SURFACE_PASCAL",
                "RELATIVE_HUMIDITY_2_METERS_ABOVE_SURFACE_PERCENT"
            ],
            label_field="TEMPERATURE_AT_SURFACE_KELVIN",
            normalize_inputs=True,
            validation_budget=ValidationBudget(
                budget_type=BudgetType.STATIC_BUDGET,
                static_budget=StaticBudget(
                    total_limit=0,
                    strata_limit=0,
                    sample_rate=0.2
                )
            ),
            loss_function=LossFunction.MEAN_SQUARED_ERROR,
            gis_joins=[
                'G3500470', 'G4200030', 'G4201090', 'G4201070', 'G4200970', 'G4200050', 'G3900030', 'G3901490',
                'G3901510',
                'G3900950', 'G3901010', 'G3901530', 'G3200050', 'G4900190', 'G3800410', 'G3800130', 'G3800210',
                'G2001650',
                'G2001690', 'G2000790', 'G2001530', 'G2000450', 'G2000750', 'G3100390', 'G3100330', 'G3101570',
                'G3100590',
                'G3101550', 'G0500290', 'G0501410', 'G0501090', 'G0500050', 'G0500010', 'G4100050', 'G1301510',
                'G1301430',
                'G1302430', 'G1301530', 'G1300610', 'G1300570', 'G1300530', 'G1302410', 'G1302470', 'G1300630',
                'G3600230',
                'G3600750', 'G3600210', 'G3600910', 'G0800890', 'G0800750', 'G0800550', 'G4000650', 'G4000610',
                'G4000410',
                'G2400130', 'G2400110', 'G2400150', 'G2801530', 'G2800830', 'G2801550', 'G2800350', 'G2800390',
                'G3000430',
                'G3000710', 'G3000830', 'G2101850', 'G2101790', 'G2101870', 'G2101910', 'G2101730', 'G2100210',
                'G2101750',
                'G3400270', 'G3400210', 'G3400190', 'G0100950', 'G0100350', 'G0101150', 'G0100410', 'G0100370',
                'G2901710',
                'G2901370', 'G2901330', 'G2901770', 'G2901390', 'G0600450', 'G0600250', 'G0601110', 'G1600190',
                'G1600390',
                'G4600450', 'G4601090', 'G4600430', 'G4600590', 'G4701250', 'G4700030', 'G4701350', 'G4700010',
                'G4700810',
                'G4700850', 'G1700770', 'G1701290', 'G1701810', 'G1700690', 'G1700710', 'G1701850', 'G1801690',
                'G1800870',
                'G1800770', 'G1801710', 'G1801730', 'G1900610', 'G1900630', 'G1900690', 'G1900990', 'G2601370',
                'G2600050',
                'G2600910', 'G2600930', 'G2601350', 'G2701490', 'G2701150', 'G2700070', 'G2700570', 'G4500670',
                'G4500590',
                'G2300310', 'G3700650', 'G3700990', 'G3700710', 'G3700670', 'G3701070', 'G4802710', 'G4800750',
                'G4804570',
                'G4803690', 'G4801390', 'G4801850', 'G4801450', 'G4804930', 'G4804590', 'G4803230', 'G4802750',
                'G4800290',
                'G2201050', 'G2201010', 'G2200990', 'G2200410', 'G2200330', 'G1200170', 'G1200870', 'G1200860',
                'G1200210',
                'G0202750'
            ]
        )
    )

    validator.validate_gis_joins(verbose=True)

    print("\n\n\n\n >>>>>>>>>>>>>>>>>>>>>>>>>> FINISHED FIRST ROUND OF VALIDATIONS <<<<<<<<<<<<<<<<<\n\n\n\n",
          flush=True)

    validator.validate_gis_joins(verbose=True)

    print("\n\n\n\n >>>>>>>>>>>>>>>>>>>>>>>>>> FINISHED SECOND ROUND OF VALIDATIONS <<<<<<<<<<<<<<<<<\n\n\n\n",
          flush=True)
