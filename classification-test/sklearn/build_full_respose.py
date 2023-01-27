# Build a complete classification response for frontend testing
import pymongo
import pandas as pd
import os
import json
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics

MONGO_USER = "root"
MONGO_PASS = "rootPass"
MONGO_HOST = "lattice-101"
MONGO_PORT = 27018
MONGO_URL = f"mongodb://{MONGO_USER}:{MONGO_PASS}@{MONGO_HOST}:{MONGO_PORT}"
DB_NAME = "sustaindb"

FEATURES_FIELDS = [
    "PRESSURE_REDUCED_TO_MSL_PASCAL",
    "VISIBILITY_AT_SURFACE_METERS",
    "VISIBILITY_AT_CLOUD_TOP_METERS",
    "WIND_GUST_SPEED_AT_SURFACE_METERS_PER_SEC",
    "PRESSURE_AT_SURFACE_PASCAL",
]

LABEL_FIELD = "CATEGORICAL_SNOW_SURFACE_BINARY"

db_connection = pymongo.MongoClient(MONGO_URL)
db = db_connection[DB_NAME]
noaa = db['noaa_nam']

gis_joins = noaa.distinct('GISJOIN')
print(f'No. of GISJOINs: {len(gis_joins)}')

count = 1
for gis_join in gis_joins:
    try:
        query = {'GISJOIN': gis_join}
        # Build projection
        projection = {"_id": 0}
        for feature in FEATURES_FIELDS:
            projection[feature] = 1
        projection[LABEL_FIELD] = 1

        single_raw_data = noaa.find(query, projection)

        # Load trained model
        model = pickle.load(open('model.pkl', 'rb'))
        features_df = pd.DataFrame(list(single_raw_data))
        scaled = MinMaxScaler(feature_range=(0, 1)).fit_transform(features_df)
        features_df = pd.DataFrame(scaled, columns=features_df.columns)

        label_df = features_df.pop(LABEL_FIELD)

        inputs_numpy = features_df.to_numpy()
        y_true = label_df.to_numpy()
        y_pred_class = model.predict(inputs_numpy)

        accuracy = metrics.accuracy_score(y_true, y_pred_class)
        print(f'Accuracy: {accuracy}')
        print(f'Percentage of 1s: {y_true.mean()}')
        print(f'Percentage of 0s: {1 - y_true.mean()}')

        null_accuracy = max(y_true.mean(), 1 - y_true.mean())
        print(f'Null accuracy: {null_accuracy}')

        confusion_matrix = metrics.confusion_matrix(y_true, y_pred_class)
        print(f'Confusion Matrix: {confusion_matrix}')

        precision = metrics.precision_score(y_true, y_pred_class)
        recall = metrics.recall_score(y_true, y_pred_class)
        print(f'Precision: {precision}')
        print(f'Recall: {recall}')

        # ROC Curves and Area Under the Curve (AUC)
        y_pred_prob = model.predict_proba(features_df)
        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred_prob)
        roc_auc_score = metrics.roc_auc_score(y_true, y_pred_prob)
        print(f'fpr: {fpr}')
        print(f'tpr: {tpr}')
        print(f'roc_auc_score: {roc_auc_score}')
    except:
        print(f'Error in {gis_join}')
