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

    tf_validator = validation.TensorflowValidator(
        "test_request_id"
        "/tmp/validation-service/saved_models",
        "Linear Regression",
        "noaa_nam",
        "COUNTY_GISJOIN",
        ["PRESSURE_AT_SURFACE_PASCAL", "RELATIVE_HUMIDITY_2_METERS_ABOVE_SURFACE_PERCENT"],
        "TEMPERATURE_AT_SURFACE_KELVIN",
        "RMSE",
        True,  # normalize
        0,
        0.0
    )

    tf_validator.validate_gis_joins_synchronous(
        [
            "G2000190",
            "G2000090",
            "G2000670",
            "G2000610",
            "G2000250",
            "G2000070",
            "G2000030",
            "G2000470"
        ]
    )


if __name__ == '__main__':
    main()
