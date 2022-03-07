#!/bin/python3

import io
import zipfile
import tensorflow as tf
import tempfile
import pandas as pd
from pprint import pprint
from pymongo import MongoClient
from sklearn.preprocessing import MinMaxScaler
import logging
import h5py
from memory_tempfile import MemoryTempfile
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


def create_and_train_model(features_df, label_df) -> tf.keras.Model:
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(2,)))
    model.add(tf.keras.layers.Dense(units=1, activation='relu'))
    model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(LEARNING_RATE))
    model.summary()

    history = model.fit(features_df, label_df, epochs=EPOCHS, validation_split=0.2)
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    pprint(hist)

    return model


def in_memory_zip():
    in_file = open("./my_model.zip", "rb")
    data = in_file.read()
    in_file.close()

    zip_file = zipfile.ZipFile(io.BytesIO(data))

    extracted_contents = {name: zip_file.read(name) for name in zip_file.namelist()}
    return extracted_contents


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

    # model: tf.keras.Model = create_and_train_model(features_df, label_df)

    # model.save("my_model.h5")

    extracted_zip = in_memory_zip()
    mem_temp = MemoryTempfile()
    temp = mem_temp.TemporaryFile()
    h5_bytes = extracted_zip["my_model.h5"]
    temp.write(h5_bytes)

    with h5py.File(temp, 'r') as h5file:
        model = tf.keras.models.load_model(h5file)
        model.summary()



    #new_model = tf.keras.models.load_model(extracted_zip['my_model.h5'])

    #new_model.summary()

    #
    # model: tf.keras.Model = tf.keras.models.load_model(extracted_zip)
    #
    # validation_results = model.evaluate(features_df, label_df, batch_size=128, return_dict=True, verbose=1)
    # info(f"Model validation results: {validation_results}")

    # cloned_model: tf.keras.Model = tf.keras.models.clone_model(model)
    # cloned_model.compile(loss=model.metrics_names['loss'])
    # validation_results = cloned_model.evaluate(features_df, label_df, batch_size=128, return_dict=True, verbose=1)
    # info(f"Cloned Model validation results: {validation_results}")


if __name__ == '__main__':
    main()
