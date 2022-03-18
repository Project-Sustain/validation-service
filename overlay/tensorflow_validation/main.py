#!/bin/python3
import numpy
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pandas
import time
import h5py
from pprint import pprint
import logging
from logging import info
from pymongo import MongoClient
from sklearn.preprocessing import MinMaxScaler

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


def create_and_train_model(features_df, label_df) -> tf.keras.Model:
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

    return model


def main():
    logging.basicConfig(level=logging.INFO)

    info("tensorflow version: {}".format(tf.__version__))

    # Load in data
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

    model: tf.keras.Model = create_and_train_model(features_df, label_df)
    model.save("my_model.h5")
    loaded_model: tf.keras.Model = tf.keras.models.load_model("my_model.h5")
    loaded_model.summary()

    #validation_results = loaded_model.evaluate(features_df, label_df, batch_size=128, return_dict=True, verbose=1)
    predictions = loaded_model.predict(features_df).numpy()

    info(f"Predictions shape: {predictions.shape}")
    pprint(predictions)
    pprint(label_df.shape)



if __name__ == "__main__":
    main()
