#!/bin/python3
import numpy
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
import h5py
from pprint import pprint
import logging
from logging import info
from pymongo import MongoClient
from bson.json_util import dumps
from sklearn.preprocessing import MinMaxScaler
from overlay.constants import username, password

# MongoDB Stuff
URI = "mongodb://{username}:{password}@lattice-100.cs.colostate.edu:27018/"
DATABASE = "sustaindb"
COLLECTION = "noaa_nam"
GIS_JOIN = "G3500170"
REGRESSION_FEATURE_FIELDS = [
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

CLASSIFICATION_FEATURE_FIELDS = [
        "PRESSURE_AT_SURFACE_PASCAL",
        "RELATIVE_HUMIDITY_2_METERS_ABOVE_SURFACE_PERCENT"
]

REGRESSION_LABEL_FIELD = "SOIL_TEMPERATURE_0_TO_01_M_BELOW_SURFACE_KELVIN"
CLASSIFICATION_LABEL_FIELD = "CATEGORICAL_RAIN_SURFACE_BINARY"

# Modeling Stuff
LEARNING_RATE = 0.01
EPOCHS = 3
BATCH_SIZE = 128


def exports():
    # Set CUDA and CUPTI paths
    os.environ['CUDA_HOME'] = '/usr/local/cuda'
    os.environ['PATH']= '/usr/local/cuda/bin:$PATH'
    os.environ['CPATH'] = '/usr/local/cuda/include:$CPATH'
    os.environ['LIBRARY_PATH'] = '/usr/local/cuda/lib64:$LIBRARY_PATH'
    os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH'
    os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda/lib64:$LD_LIBRARY_PATH'


def create_and_train_regression_model(features_df, label_df) -> tf.keras.Model:
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=features_df.shape[1],))
    model.add(tf.keras.layers.Dense(units=32, activation="relu", name="first_layer"))
    model.add(tf.keras.layers.Dense(units=16, activation="relu", name="second_layer"))
    model.add(tf.keras.layers.Dense(units=1, activation="relu", name="third_layer"))
    model.compile(loss="mean_squared_error", optimizer=tf.keras.optimizers.Adam(LEARNING_RATE))
    model.summary()

    history = model.fit(features_df, label_df, epochs=EPOCHS, validation_split=0.2)
    hist = pd.DataFrame(history.history)
    hist["epoch"] = history.epoch
    pprint(hist)

    return model


def get_data_for_regression_model() -> (pd.DataFrame, pd.DataFrame):
    # Load in data
    client = MongoClient(URI)
    database = client["sustaindb"]
    collection = database["noaa_nam"]
    match = {"GISJOIN": GIS_JOIN}
    projection = {"_id": 0, REGRESSION_LABEL_FIELD: 1}
    for feature_field in REGRESSION_FEATURE_FIELDS:
        projection[feature_field] = 1
    documents = collection.find(match, projection)
    features_df = pd.DataFrame(list(documents))
    scaled = MinMaxScaler(feature_range=(0, 1)).fit_transform(features_df)
    features_df = pd.DataFrame(scaled, columns=features_df.columns)
    label_df = features_df.pop(REGRESSION_LABEL_FIELD)

    return features_df, label_df


def load_from_disk() -> (pd.DataFrame, pd.DataFrame):
    path_to_noaa_csv: str = "/home/inf0rmatiker/noaa_nam_normalized.csv"
    all_df: pd.DataFrame = pd.read_csv(path_to_noaa_csv, header=0)
    features: pd.DataFrame = all_df[REGRESSION_FEATURE_FIELDS]
    labels: pd.DataFrame = all_df[REGRESSION_LABEL_FIELD]

    return features, labels


def get_data_for_classification_model() -> (pd.DataFrame, pd.DataFrame):
    # Load in data
    client = MongoClient(URI)
    database = client["sustaindb"]
    collection = database["noaa_nam"]
    match_with = {"GISJOIN": GIS_JOIN, CLASSIFICATION_LABEL_FIELD: 1}
    projection = {"_id": 0, CLASSIFICATION_LABEL_FIELD: 1}
    for feature_field in CLASSIFICATION_FEATURE_FIELDS:
        projection[feature_field] = 1
    documents_with = collection.find(match_with, projection)
    features_df_with = pd.DataFrame(list(documents_with))

    match_without = {"GISJOIN": GIS_JOIN, CLASSIFICATION_LABEL_FIELD: 0}
    documents_without = collection.find(match_without, projection).limit(features_df_with.shape[0])
    features_df_without = pd.DataFrame(list(documents_without))

    all_features_df = features_df_with.append(features_df_without)
    label_df = all_features_df.pop(CLASSIFICATION_LABEL_FIELD)
    pprint(all_features_df)

    return all_features_df, label_df


def main():
    logging.basicConfig(level=logging.INFO)
    info("tensorflow version: {}".format(tf.__version__))
    model_type = ""  # change this to run either regression or classification

    features_df, label_df = load_from_disk()
    regression_model: tf.keras.Model = create_and_train_regression_model(features_df, label_df)
    regression_model.save("/home/inf0rmatiker/my_regression_model.h5")
    loaded_model: tf.keras.Model = tf.keras.models.load_model("/home/inf0rmatiker/my_regression_model.h5")
    loaded_model.summary()

    # validation_results = loaded_model.evaluate(features_df, label_df, batch_size=128, return_dict=True, verbose=1)
    # y_pred = loaded_model.predict(features_df)

    # info(f"Predictions shape: {y_pred.shape}")
    # pprint(y_pred)

    # y_true = np.array(label_df).reshape(-1, 1)
    # pprint(y_true)

    # mse = tf.keras.losses.MeanSquaredError()
    # loss = mse(y_true, y_pred).numpy()
    #
    # loss = np.mean(np.abs(y_true - y_pred), axis=0)[0]

    # info(f"Loss: {loss}")

    # input_variance = y_true.var()
    # absolute_error_variance = np.absolute(y_pred - y_true).var()
    # info(f"Absolute variance error: {absolute_error_variance}, input variance: {input_variance}")

if __name__ == "__main__":
    main()
