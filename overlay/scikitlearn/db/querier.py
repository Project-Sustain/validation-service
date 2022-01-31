import pymongo


class Querier:
    def __init__(self, mongo_url: str, db_name: str):
        self.db_connection = pymongo.MongoClient(mongo_url)
        self.db = self.db_connection[db_name]

    def query(self, collection_name: str, gis_join: str, fields):
        collection = self.db[collection_name]
        client_query = {fields: gis_join}
        query_results = list(collection.find(client_query))
        return query_results
