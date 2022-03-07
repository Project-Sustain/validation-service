#!/bin/python3

import numpy
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pandas
import time
from pprint import pprint
#from sklearn.model_selection import train_test_split
#from tensorflow.keras import layers
#from tensorflow.keras.layers.experimental import preprocessing
from pymongo import MongoClient
from sklearn.preprocessing import normalize, MinMaxScaler
import logging
from logging import info, error

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


def create_and_train_model(features_df, label_df):
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(1,)))
    model.add(tf.keras.layers.Dense(units=1, activation='relu'))
    model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(LEARNING_RATE))
    model.summary()

    history = model.fit(features_df, label_df, epochs=EPOCHS, validation_split=0.2)
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    pprint(hist)


def main():
    logging.basicConfig(level=logging.INFO)
    print("tensorflow version: {}".format(tf.__version__))



    # Load in data
    client = MongoClient(URI)
    database = client["sustaindb"]
    collection = database["noaa_nam"]
    documents = collection.find(
        {"GISJOIN": GIS_JOIN},
        {"_id": 0, "PRESSURE_AT_SURFACE_PASCAL": 1, "RELATIVE_HUMIDITY_2_METERS_ABOVE_SURFACE_PERCENT": 1,
         "TEMPERATURE_AT_SURFACE_KELVIN": 1}
    )
    features_df = pd.DataFrame(list(documents))
    scaled = MinMaxScaler(feature_range=(0, 1)).fit_transform(features_df)
    features_df = pd.DataFrame(scaled, columns=features_df.columns)
    label_df = features_df.pop(LABEL_FIELD)

    create_and_train_model(features_df, label_df)

    # validation_results = model.evaluate(features_df, label_df, batch_size=128, return_dict=True, verbose=0)
    # info(f"Model validation results: {validation_results}")


if __name__ == '__main__':
    main()
