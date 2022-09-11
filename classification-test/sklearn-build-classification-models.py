import numpy as np
import pandas as pd
import pymongo
import os
import pickle
import random
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

CLASSIFICATION_FEATURES_FIELDS = [
    "PRESSURE_REDUCED_TO_MSL_PASCAL",
    "VISIBILITY_AT_SURFACE_METERS",
    "VISIBILITY_AT_CLOUD_TOP_METERS",
    "WIND_GUST_SPEED_AT_SURFACE_METERS_PER_SEC",
    "PRESSURE_AT_SURFACE_PASCAL",
    # "TEMPERATURE_AT_SURFACE_KELVIN",
    # "DEWPOINT_TEMPERATURE_2_METERS_ABOVE_SURFACE_KELVIN",
    # "RELATIVE_HUMIDITY_2_METERS_ABOVE_SURFACE_PERCENT",
    # "ALBEDO_PERCENT",
    # "TOTAL_CLOUD_COVER_PERCENT"
]

CLASSIFICATION_LABEL_FIELD = [
    "CATEGORICAL_SNOW_SURFACE_BINARY"
]

QUERY = False 

HOST = "lattice-100"
MONGO_USERNAME = os.environ["ROOT_MONGO_USER"]
MONGO_PASSWORD = os.environ["ROOT_MONGO_PASS"]

mongo_url = f"mongodb://{MONGO_USERNAME}:{MONGO_PASSWORD}@{HOST}:27018"
print(f"[INFO]: Mongo URL: {mongo_url}")
sustain_client = pymongo.MongoClient(mongo_url)
sustain_db = sustain_client['sustaindb']

if QUERY:
    raw_data = sustain_db['noaa_nam'].find({'GISJOIN': 'G0600470'})
    df = pd.DataFrame(list(raw_data))
    pickle.dump(df, open('pickles/noaa_df.pkl', 'wb'))
    print('INFO: DF created')
else:
    df = pickle.load(open('pickles/noaa_df.pkl', 'rb'))
    print('INFO: DF loaded')

# available on lattice-176
# path_to_noaa_csv: str = "~/noaa_nam_normalized.csv"
# all_df: pd.DataFrame = pd.read_csv(path_to_noaa_csv, header=0)

X_df: pd.DataFrame = df[CLASSIFICATION_FEATURES_FIELDS]
y_df: pd.DataFrame = df[CLASSIFICATION_LABEL_FIELD]

X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=0.25)

print(f'X_train: {X_train.shape}')
print(f'y_train: {y_train.shape}')
print(f'X_test: {X_test.shape}')
print(f'y_test: {y_test.shape}')

print('INFO: Train and Test sets created')

dtree_model = DecisionTreeClassifier(max_depth=2).fit(X_train, y_train)
print('INFO: Model created')

dtree_predictions = dtree_model.predict(X_test)
pickle.dump(dtree_model, open('./pickles/classification_model.pkl', 'wb'))
print("[INFO] Model serialized to file")
