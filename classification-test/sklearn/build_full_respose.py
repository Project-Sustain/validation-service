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

LABEL_FIELD = [
    "CATEGORICAL_SNOW_SURFACE_BINARY"
]

db_connection = pymongo.MongoClient(MONGO_URL)
db = db_connection[DB_NAME]
noaa = db['noaa_nam']

single_raw_data = noaa.find({'GISJOIN': 'G0600470'})
df = pd.DataFrame(list(single_raw_data))
print("DF:")
print(df)

print('---------------------')
gis_joins = noaa.distinct('GISJOIN')
print(f'No. of GISJOINs: {len(gis_joins)}')
print('---------------------\n')

# Load trained model
model = pickle.load(open('model.pkl', 'rb'))
features_df = pd.DataFrame(list(single_raw_data))
scaled = MinMaxScaler(feature_range=(0, 1)).fit_transform(features_df)
features_df = pd.DataFrame(scaled, columns=features_df.columns)

label_df = features_df.pop(LABEL_FIELD)
features_df = features_df[FEATURES_FIELDS]

inputs_numpy = features_df.to_numpy()
y_true = label_df.to_numpy()
y_pred_class = model.predict(inputs_numpy)

accuracy = metrics.accuracy_score(y_true, y_pred_class)
print(f'Accuracy: {accuracy}')
