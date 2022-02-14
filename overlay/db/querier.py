import pymongo
from pymongo import cursor
from logging import info


class Querier:

    def __init__(self, mongo_uri: str, db_name: str):
        self.mongo_uri = mongo_uri
        self.db_name = db_name
        self.db_connection = pymongo.MongoClient(self.mongo_uri, readPreference="nearest")
        self.db = self.db_connection[self.db_name]

    # Executes a spatial query on a MongoDB collection, projecting it to return only the features and label values.
    def spatial_query(self,
                      collection_name: str,
                      spatial_key: str,
                      spatial_value: str,
                      features: list,
                      label: str,
                      limit=0,
                      sample_rate=0.0) -> cursor.Cursor:

        collection = self.db[collection_name]
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


if __name__ == "__main__":
    pass
    # sustaindb = Querier(f'{DB_HOST}:{DB_PORT}', DB_NAME)
    # results = sustaindb.query("county_median_age", "G0100150")
    # print(results)
