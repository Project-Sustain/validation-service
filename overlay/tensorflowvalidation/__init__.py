import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
from pprint import pprint
from pymongo import MongoClient
from logging import info, error


URI = "mongodb://localhost:27018"


def validate_model(job_id, path, model_type, database, collection, label_field, validation_metric, feature_fields, gis_join):
    client = MongoClient(URI)
    database = client[database]
    collection = database[collection]

    documents = collection.find({'COUNTY_GISJOIN': gis_join}, {'_id': 0, feature_fields[0]: 1, label_field: 1})
    features = []
    labels = []
    num_processed = 0
    for document in documents:
        features.append(document[feature_fields[0]])
        labels.append(document[label_field])
        num_processed += 1

        if num_processed % 1000 == 0:
            info(f"Processed {num_processed} documents...")

    np_features = np.array(features)
    np_labels = np.array(labels)

    info(f"np_features shape: {np_features.shape}")
    info(f"np_labels shape: {np_labels.shape}")

    normalized_features = tf.keras.utils.normalize(
        np_features, axis=-1, order=2
    ).transpose()

    normalized_labels = tf.keras.utils.normalize(
        np_labels, axis=-1, order=2
    ).transpose()

    pprint(normalized_features)
    info(f"normalized_features shape: {normalized_features.shape}")
    info(f"normalized_labels shape: {normalized_labels.shape}")

    # Reload model
    new_model = tf.keras.models.load_model(f"{path}/{job_id}")

    # Check its architecture
    new_model.summary()

    new_results = new_model.evaluate(normalized_features, normalized_labels, batch_size=128)
    info(f"Test loss, test acc: {new_results}")

    client.close()
