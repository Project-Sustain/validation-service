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

MODEL_PATH = "../../testing/test_models/tensorflow/my_model.zip"
URI = "mongodb://lattice-100:27018/"
DATABASE = "sustaindb"
COLLECTION = "noaa_nam"

# Modeling Stuff
LEARNING_RATE = 0.001
EPOCHS = 3
BATCH_SIZE = 32

def main():
    logging.basicConfig(level=logging.INFO)
    print("tensorflow version: {}".format(tf.__version__))

    info(f"Loading Tensorflow model from {MODEL_PATH}")
    model: tf.keras.Model = tf.keras.models.load_model(MODEL_PATH)
    model.summary()

    gis_join = "G3500170"

    feature_fields = [
        "PRESSURE_AT_SURFACE_PASCAL",
        "RELATIVE_HUMIDITY_2_METERS_ABOVE_SURFACE_PERCENT"
    ]
    label_field = "TEMPERATURE_AT_SURFACE_KELVIN"

    db_connection = MongoClient(URI)
    documents = db_connection["sustaindb"]["noaa_nam"].find(
        {"GISJOIN": gis_join},
        {"_id": 0, "PRESSURE_AT_SURFACE_PASCAL": 1, "RELATIVE_HUMIDITY_2_METERS_ABOVE_SURFACE_PERCENT": 1, "TEMPERATURE_AT_SURFACE_KELVIN": 1}
    )

    features_df = pd.DataFrame(list(documents))

    scaled = MinMaxScaler(feature_range=(0, 1)).fit_transform(features_df)
    features_df = pd.DataFrame(scaled, columns=features_df.columns)

    label_df = features_df.pop(label_field)
    validation_results = model.evaluate(features_df, label_df, batch_size=128, return_dict=True, verbose=0)
    info(f"Model validation results: {validation_results}")


if __name__ == '__main__':
    main()
