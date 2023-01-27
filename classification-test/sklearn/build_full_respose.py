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

db_connection = pymongo.MongoClient(MONGO_URL)
db = db_connection[DB_NAME]
noaa = db['noaa_nam']

single_raw_data = noaa.find({'GISJOIN': 'G0600470'})
df = pd.DataFrame(list(single_raw_data))
print("DF:")
print(df)

print('---------------------')
gis_joins = noaa.distinct('GISJOIN')
print(len(gis_joins))

