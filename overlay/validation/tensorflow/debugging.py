import os
import json
import pandas as pd
import numpy as np
import gc
import tensorflow as tf
import time
from pymongo import MongoClient
from sklearn.preprocessing import MinMaxScaler
from math import sqrt


class Timer:

    def __init__(self, func=time.perf_counter):
        self.elapsed = 0.0
        self._func = func
        self._start = None

    # starting the module
    def start(self):
        if self._start is not None:
            raise RuntimeError('Already started')
        self._start = self._func()

    # stopping the timer
    def stop(self):
        if self._start is None:
            raise RuntimeError('Not started')
        end = self._func()
        self.elapsed += end - self._start
        self._start = None

    # resetting the timer
    def reset(self):
        self.elapsed = 0.0

    def running(self):
        return self._start is not None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


def run(gis_join):
    print(f"GISJOIN: {gis_join}")

    profiler: Timer = Timer()
    profiler.start()

    features = [
        "PRESSURE_REDUCED_TO_MSL_PASCAL",
        "VISIBILITY_AT_SURFACE_METERS",
        "VISIBILITY_AT_CLOUD_TOP_METERS",
        "WIND_GUST_SPEED_AT_SURFACE_METERS_PER_SEC",
        "PRESSURE_AT_SURFACE_PASCAL",
        "TEMPERATURE_AT_SURFACE_KELVIN",
        "DEWPOINT_TEMPERATURE_2_METERS_ABOVE_SURFACE_KELVIN",
        "RELATIVE_HUMIDITY_2_METERS_ABOVE_SURFACE_PERCENT",
        "ALBEDO_PERCENT",
        "TOTAL_CLOUD_COVER_PERCENT"
    ]
    label = "SOIL_TEMPERATURE_0_TO_01_M_BELOW_SURFACE_KELVIN"
    model_path = "../../../testing/test_models/tensorflow/neural_network/hdf5/model.h5"
    model: tf.keras.Model = tf.keras.models.load_model(model_path)

    profiler.stop()
    print(f">>> Loading model: {profiler.elapsed} sec")
    profiler.reset()

    profiler.start()
    client = MongoClient("mongodb://localhost:27017")
    db = client["sustaindb"]
    coll = db["noaa_nam"]
    query = {"GISJOIN": gis_join}
    # Build projection
    projection = {"_id": 0}
    for feature in features:
        projection[feature] = 1
    projection[label] = 1

    documents = coll.find(query, projection)

    features_df = pd.DataFrame(list(documents))
    client.close()

    profiler.stop()
    #print(f"Size of DF: {len(features_df.index)}")
    print(f">>> Loading data into pandas df: {profiler.elapsed} sec")
    profiler.reset()

    profiler.start()

    scaled = MinMaxScaler(feature_range=(0, 1)).fit_transform(features_df)
    features_df = pd.DataFrame(scaled, columns=features_df.columns)
    #print(f"Normalized Pandas DataFrame")

    label_df = features_df.pop(label)

    profiler.stop()
    print(f">>> Normalizing pandas df: {profiler.elapsed} sec")
    profiler.reset()

    profiler.start()

    # Get predictions
    y_pred = model(features_df.values.astype(np.float32), verbose=0)

    profiler.stop()
    print(f">>> Getting predicted values from model: {profiler.elapsed} sec")
    profiler.reset()

    profiler.start()

    # Use labels and predictions to evaluate the model
    y_true = np.array(label_df).reshape(-1, 1)

    profiler.stop()
    print(f">>> Reshaping label_df as numpy column: {profiler.elapsed} sec")
    profiler.reset()

    profiler.start()

    squared_residuals = np.square(y_true - y_pred)
    m = np.mean(squared_residuals, axis=0)[0]
    loss = m
    s = (np.var(squared_residuals, axis=0, ddof=0) * squared_residuals.shape[0])[0]

    profiler.stop()
    print(f">>> Getting loss criterion from residuals: {profiler.elapsed} sec")
    profiler.reset()

    #print(loss)


if __name__ == "__main__":
    run("G4801090")
