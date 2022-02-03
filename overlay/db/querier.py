import pymongo

from overlay.constants import DB_HOST, DB_PORT, DB_NAME


class Querier:
    def __init__(self, mongo_url: str, db_name: str):
        self.db_connection = pymongo.MongoClient(mongo_url)
        self.db = self.db_connection[db_name]

    def query(self, collection_name: str, gis_join: str):
        collection = self.db[collection_name]
        client_query = {"GISJOIN": gis_join}
        query_results = list(collection.find(client_query))
        return query_results


if __name__ == "__main__":
    sustaindb = Querier(f'{DB_HOST}:{DB_PORT}', DB_NAME)
    results = sustaindb.query("county_median_age", "G0100150")
    print(results)
