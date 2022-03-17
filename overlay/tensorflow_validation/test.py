import os
import pymongo
import tensorflow as tf
import pandas as pd
import logging
import time

from logging import info, error
from pprint import pprint
from pymongo import cursor, ReadPreference, MongoClient
from sklearn.preprocessing import MinMaxScaler
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed


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


# MongoDB Stuff
URI = "mongodb://lattice-100:27018/"
DATABASE = "sustaindb"
COLLECTION = "noaa_nam"
GIS_JOIN = "G3500170"
FEATURE_FIELDS = [
        "PRESSURE_AT_SURFACE_PASCAL",
        "RELATIVE_HUMIDITY_2_METERS_ABOVE_SURFACE_PERCENT"
]
LABEL_FIELD = "TEMPERATURE_AT_SURFACE_KELVIN"

# Modeling Stuff
LEARNING_RATE = 0.001
EPOCHS = 3
BATCH_SIZE = 32


def test_func_synchronous(model_id: int):

    # Pull in data from MongoDB into Pandas DataFrame
    client = MongoClient(URI)
    database = client["sustaindb"]
    collection = database["noaa_nam"]
    match = {"GISJOIN": GIS_JOIN}
    projection = {"_id": 0, LABEL_FIELD: 1}
    for feature_field in FEATURE_FIELDS:
        projection[feature_field] = 1
    documents = collection.find(match, projection)
    features_df = pd.DataFrame(list(documents))
    scaled = MinMaxScaler(feature_range=(0, 1)).fit_transform(features_df)
    features_df = pd.DataFrame(scaled, columns=features_df.columns)
    label_df = features_df.pop(LABEL_FIELD)

    # Create and train Keras model
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(2,)))
    model.add(tf.keras.layers.Dense(units=16, activation="relu", name="first_layer"))
    model.add(tf.keras.layers.Dense(units=4, activation="relu", name="second_layer"))
    model.compile(loss="mean_squared_error", optimizer=tf.keras.optimizers.Adam(LEARNING_RATE))
    model.summary()

    history = model.fit(features_df, label_df, epochs=EPOCHS, validation_split=0.2)
    hist = pd.DataFrame(history.history)
    hist["epoch"] = history.epoch
    pprint(hist)

    model.save(f"my_model_{model_id}.h5")
    loaded_model: tf.keras.Model = tf.keras.models.load_model(f"my_model_{model_id}.h5")
    loaded_model.summary()

    validation_results = loaded_model.evaluate(features_df, label_df, batch_size=128, return_dict=True, verbose=1)
    info(f"Model validation results: {validation_results}")


def main():
    logging.basicConfig(level=logging.INFO)
    profiler: Timer = Timer()

    profiler.start()
    for i in range(3):
        test_func_synchronous(i)
    profiler.stop()
    info(f"Time elapsed: {profiler.elapsed}")


if __name__ == "__main__":
    main()
