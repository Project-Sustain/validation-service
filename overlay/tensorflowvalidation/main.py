#!/bin/python3

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
from pprint import pprint
#from sklearn.model_selection import train_test_split
#from tensorflow.keras import layers
#from tensorflow.keras.layers.experimental import preprocessing
import tensorflow_io as tfio
from pymongo import MongoClient

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
    print("tensorflow-io version: {}".format(tfio.__version__))

    client = MongoClient(URI)
    database = client[DATABASE]
    collection = database[COLLECTION]

    feature = 'TEMPERATURE_AT_SURFACE_KELVIN'
    label = 'TEMPERATURE_TROPOPAUSE_KELVIN'

    documents = collection.find({'GISJOIN': 'G4802970'}, {'_id': 0, feature: 1, label: 1})
    features = []
    labels = []
    num_processed = 0
    for document in documents:
        features.append(document[feature])
        labels.append(document[label])
        num_processed += 1

        if num_processed % 1000 == 0:
            print(f"Processed {num_processed} documents...")

    np_features = np.array(features)
    np_labels = np.array(labels)

    print(f"np_features shape: {np_features.shape}")
    print(f"np_labels shape: {np_labels.shape}")

    normalized_features = tf.keras.utils.normalize(
        np_features, axis=-1, order=2
    ).transpose()

    normalized_labels = tf.keras.utils.normalize(
        np_labels, axis=-1, order=2
    ).transpose()

    pprint(normalized_features)
    print(f"normalized_features shape: {normalized_features.shape}")
    print(f"normalized_labels shape: {normalized_labels.shape}")

    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(1,)))
    model.add(tf.keras.layers.Dense(units=1, activation='relu'))
    model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(LEARNING_RATE))
    model.summary()

    history = model.fit(normalized_features, normalized_labels, epochs=EPOCHS, validation_split=0.2)
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    pprint(hist)

    results = model.evaluate(normalized_features, normalized_labels, batch_size=128)
    print("test loss, test acc:", results)

    # Save model
    model.save('saved_model/my_model')

    # Reload model
    new_model = tf.keras.models.load_model('saved_model/my_model')

    # Check its architecture
    new_model.summary()

    new_results = new_model.evaluate(normalized_features, normalized_labels, batch_size=128)
    print("RELOADED test loss, test acc:", new_results)

    # first = np.array(np_features[:1])
    #
    # with np.printoptions(precision=2, suppress=True):
    #     print('First example:', first)
    #     print()
    #     print('Normalized:', normalizer(first).numpy())

    client.close()


if __name__ == '__main__':
    main()
