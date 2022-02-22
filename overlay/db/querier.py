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
        self.mongo_uri = f"mongodb://{mongo_host}:{mongo_port}"
        self.replica_set_status = locality.get_replica_set_status()
        self.db_name = db_name
        self.read_preference = read_preference
        self.read_concern = read_concern
        self.db_connection = pymongo.MongoClient(self.mongo_uri, readPreference=self.read_preference)
        self.db = self.db_connection[self.db_name]

    # Executes a spatial query on a MongoDB collection, projecting it to return only the features and label values.
    def spatial_query(self,
                      collection_name: str,
                      spatial_key: str,
                      spatial_value: str,
                      features: list,
                      label: str,
                      limit: int,
                      sample_rate: float) -> cursor.Cursor:

        collection = self.db[collection_name]
        if self.read_preference == "inferred":
            info("\"inferred\" read preference specified by request, using ")
            preference: ReadPreference = inferred_read_preference_from_rs_status(self.read_preference)
            collection = collection.with_options(read_preference=preference)

        query = {spatial_key: spatial_value}

        if sample_rate > 0.0:
            info(f"Sampling GISJOIN {spatial_value} documents with a rate of {sample_rate}")
            query["$sampleRate"] = sample_rate

        # Build projection
        projection = {"_id": 0}
        for feature in features:
            projection[feature] = 1
        projection[label] = 1

        if limit > 0:
            info(f"Limiting GISJOIN {spatial_value} query to {limit} records")
            return collection.find(query, projection).limit(limit)
        else:
            return collection.find(query, projection)  # Just find all that match

    def close(self):
        self.db_connection.close()

    def __repr__(self):
        return f"Querier: mongo_uri={self.mongo_uri}, db_name={self.db_name}"


def inferred_read_preference_from_rs_status(status: str = "PRIMARY") -> ReadPreference:
    if status == "PRIMARY":
        return ReadPreference.PRIMARY_PREFERRED
    elif status == "SECONDARY":
        return ReadPreference.SECONDARY_PREFERRED
    else:
        return ReadPreference.NEAREST


if __name__ == "__main__":
    pass
    # sustaindb = Querier(f'{DB_HOST}:{DB_PORT}', DB_NAME)
    # results = sustaindb.query("county_median_age", "G0100150")
    # print(results)
