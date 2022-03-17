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


def train_and_evaluate(model_id: int):

    # Pull in data from MongoDB into Pandas DataFrame
    client = MongoClient(URI, connect=False)
    database = client["sustaindb"]
    collection = database["noaa_nam"]
    match = {"GISJOIN": "G3500170"}
    projection = {"_id": 0, "TEMPERATURE_AT_SURFACE_KELVIN": 1, "PRESSURE_AT_SURFACE_PASCAL": 1, "RELATIVE_HUMIDITY_2_METERS_ABOVE_SURFACE_PERCENT": 1}
    documents = collection.find(match, projection)
    features_df = pd.DataFrame(list(documents))
    client.close()
    scaled = MinMaxScaler(feature_range=(0, 1)).fit_transform(features_df)
    features_df = pd.DataFrame(scaled, columns=features_df.columns)
    label_df = features_df.pop("TEMPERATURE_AT_SURFACE_KELVIN")

    # Create and train Keras model
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(2,)))
    model.add(tf.keras.layers.Dense(units=16, activation="relu", name="first_layer"))
    model.add(tf.keras.layers.Dense(units=4, activation="relu", name="second_layer"))
    model.compile(loss="mean_squared_error", optimizer=tf.keras.optimizers.Adam(0.001))
    model.summary()

    history = model.fit(features_df, label_df, epochs=3, validation_split=0.2)
    hist = pd.DataFrame(history.history)
    hist["epoch"] = history.epoch
    pprint(hist)

    model.save(f"my_model_{model_id}.h5")
    loaded_model: tf.keras.Model = tf.keras.models.load_model(f"my_model_{model_id}.h5")
    loaded_model.summary()

    validation_results = loaded_model.evaluate(features_df, label_df, batch_size=128, return_dict=True, verbose=1)
    info(f"Model validation results: {validation_results}")


def test_synchronous():
    for i in range(3):
        train_and_evaluate(i)


def test_multithreaded():
    # Iterate over all gis_joins and submit them for validation to the thread pool executor
    executors_list = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        for i in range(3):
            executors_list.append(executor.submit(train_and_evaluate, i))

    # Wait on all tasks to finish -- Iterate over completed tasks, get their result, and log/append to responses
    for future in as_completed(executors_list):
        future.result()


def test_multiprocessed():
    # Iterate over all gis_joins and submit them for validation to the thread pool executor
    executors_list = []
    with ProcessPoolExecutor(max_workers=10) as executor:
        for i in range(3):
            executors_list.append(executor.submit(train_and_evaluate, i))

    # Wait on all tasks to finish -- Iterate over completed tasks, get their result, and log/append to responses
    for future in as_completed(executors_list):
        future.result()


def main():
    logging.basicConfig(level=logging.INFO)
    profiler: Timer = Timer()

    profiler.start()
    test_synchronous()
    profiler.stop()
    info(f"Single-threaded time elapsed: {profiler.elapsed}")
    profiler.reset()

    profiler.start()
    test_multithreaded()
    profiler.stop()
    info(f"Multi-threaded time elapsed: {profiler.elapsed}")
    profiler.reset()

    profiler.start()
    test_multiprocessed()
    profiler.stop()
    info(f"Multi-processed time elapsed: {profiler.elapsed}")
    profiler.reset()


if __name__ == "__main__":
    main()
