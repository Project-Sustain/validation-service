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

import validation

# MongoDB Stuff
URI = "mongodb://lattice-100:27018/"
DATABASE = "sustaindb"
COLLECTION = "noaa_nam"

# Modeling Stuff
LEARNING_RATE = 0.001
EPOCHS = 3
BATCH_SIZE = 32


def main():
    print("tensorflow version: {}".format(tf.__version__))

    m = 2

    features = ['PRESSURE_AT_SURFACE_PASCAL', 'RELATIVE_HUMIDITY_2_METERS_ABOVE_SURFACE_PERCENT']
    label = 'TEMPERATURE_AT_SURFACE_KELVIN'
    projection = {
        "_id": 0,
    }
    for feature in features:
        projection[feature] = 1
    projection[label] = 1

    pprint(projection)

    client = MongoClient(URI)
    database = client[DATABASE]
    collection = database[COLLECTION]
    documents = collection.find({'COUNTY_GISJOIN': 'G2000010'}, projection)

    validation.validate_model(
        "saved_model",
        "my_model",
        "Linear Regression",
        documents,
        features,
        label,
        "RMSE",
        True
    )

    client.close()


if __name__ == '__main__':
    main()
