import torch
import os
import json
import pandas as pd
import numpy as np
import gc
import pymongo
from pymongo import MongoClient
from sklearn.preprocessing import MinMaxScaler
from math import sqrt
from torch.utils.data import Dataset, DataLoader


def run(gis_join):

    features = [
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
    label = "SOIL_TEMPERATURE_0_TO_01_M_BELOW_SURFACE_KELVIN"
    model_path = "../../../testing/test_models/pytorch/neural_network/model.pt"
    model = torch.jit.load(model_path)
    model.eval()

    client = MongoClient("mongodb://localhost:27017")
    db = client["sustaindb"]
    coll = db["noaa_nam"]
    query = {"GISJOIN": gis_join}
    # Build projection
    projection = {"_id": 0}
    for feature in features:
        projection[feature] = 1
    projection[label] = 1

    documents = coll.find(query, projection)

    features_df = pd.DataFrame(list(documents))
    client.close()

    scaled = MinMaxScaler(feature_range=(0, 1)).fit_transform(features_df)
    features_df = pd.DataFrame(scaled, columns=features_df.columns)
    print(f"Normalized Pandas DataFrame")

    label_df = features_df.pop(label)

    inputs_numpy = features_df.values.astype(np.float32)
    y_true_numpy = label_df.values.astype(np.float32)
    inputs: torch.Tensor = torch.from_numpy(inputs_numpy)
    y_true: torch.Tensor = torch.from_numpy(y_true_numpy)
    y_true = y_true.view(y_true.shape[0], 1)  # convert y to a column vector

    n_samples, n_features = inputs.shape
    print(f'n_samples: {n_samples}, n_features: {n_features}')

    with torch.no_grad():
        criterion = torch.nn.MSELoss()
        y_predicted = model(inputs)
        # y_predicted_numpy = y_predicted.detach().numpy()
        loss = criterion(y_predicted, y_true)
        # squared_residuals = np.square(y_predicted_numpy - y_true_numpy)
        print(loss)


def main():
    run("G5501170")


if __name__ == "__main__":
    main()