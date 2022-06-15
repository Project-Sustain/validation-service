import urllib
import os
import pymongo
from pymongo import cursor, ReadPreference
from logging import info

from overlay.db import locality


class Querier:

    def __init__(self,
                 mongo_host: str = "localhost",
                 mongo_port: int = 27018,
                 db_name: str = "sustaindb",
                 read_preference: str = "nearest",
                 read_concern: str = "local"):

        self.mongo_port = mongo_port
        self.mongo_host = mongo_host
        # authorization --> don't commit to this branch

        # username = urllib.parse.quote_plus(os.environ.get('READ_MONGO_USER'))
        # password = urllib.parse.quote_plus(os.environ.get('READ_MONGO_PASS'))

        # username = "root"
        # password = "rootPass"
        #
        #
        # self.mongo_uri = f"mongodb://{username}:{password}@{mongo_host}:{mongo_port}"

        # end authorization

        self.mongo_uri = f"mongodb://{mongo_host}:{mongo_port}"  # Old URI pre authorization

        self.replica_set_status = locality.get_replica_set_status()
        self.db_name = db_name
        self.read_preference = read_preference
        self.read_concern = read_concern
        if mongo_port == 27018:
            self.db_connection = pymongo.MongoClient(self.mongo_uri, readPreference=self.read_preference)
        else:
            self.db_connection = pymongo.MongoClient(self.mongo_uri)
        self.db = self.db_connection[self.db_name]

    # Executes a spatial query on a MongoDB collection, projecting it to return only the features and label values.
    def spatial_query(self,
                      collection_name: str,
                      gis_join: str,
                      features: list,
                      label: str,
                      limit: int,
                      sample_rate: float) -> cursor.Cursor:

        collection = self.db[collection_name]

        if self.mongo_port == 27018:
            if self.read_preference == "nearest":
                preference: ReadPreference = ReadPreference.NEAREST
            else:
                preference: ReadPreference = ReadPreference.PRIMARY
            collection = collection.with_options(read_preference=preference)

        query = {"GISJOIN": gis_join}

        if sample_rate > 0.0:
            info(f"Sampling GISJOIN {gis_join} documents with a rate of {sample_rate}")
            query["$sampleRate"] = sample_rate

        # Build projection
        projection = {"_id": 0}
        for feature in features:
            projection[feature] = 1
        projection[label] = 1

        if limit > 0:
            info(f"Limiting GISJOIN {gis_join} query to {limit} records")
            sample = {"size": limit}
            return collection.aggregate(
                [
                    {"$match": query},
                    {"$project": projection},
                    {"$sample": sample},
                ],
                allowDiskUse=True
            )
        else:
            return collection.find(query, projection)  # Just find all that match

    def close(self):
        self.db_connection.close()

    def __repr__(self):
        return f"Querier: mongo_uri={self.mongo_uri}, db_name={self.db_name}"


if __name__ == "__main__":
    pass
    # sustaindb = Querier(f'{DB_HOST}:{DB_PORT}', DB_NAME)
    # results = sustaindb.query("county_median_age", "G0100150")
    # print(results)
