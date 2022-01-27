import json
import logging
from logging import info, error
from pymongo import MongoClient


def get_gis_join_locations():
    resources_dir = 'resources'
    county_gis_joins = load_gis_joins(resources_dir)
    info(f"Loaded in county GISJOIN list of size {len(county_gis_joins)}, retrieving chunk locations from MongoDB...")

    mongo_client = MongoClient("mongodb://lattice-100:27018")
    db = mongo_client["sustaindb"]
    collection = db["noaa_nam"]
    for gis_join in county_gis_joins.keys():
        explained_query = collection.find({"COUNTY_GISJOIN": gis_join}).explain()
        winning_plan = explained_query["queryPlanner"]["winningPlan"]
        shards = winning_plan["shards"]

        if len(shards) == 1:
            winning_shard = shards[0]
            winning_shard_name = winning_shard["shardName"]
            winning_shard_server = winning_shard["serverInfo"]["host"]
            info(f"Found GISJOIN={gis_join} at shard={winning_shard_name}, server={winning_shard_server}")
            county_gis_joins[gis_join].append(winning_shard_server)

    return county_gis_joins


def load_gis_joins(resources_dir):
    county_gis_joins = {}
    gis_join_filename = f"{resources_dir}/gis_joins.json"
    with open(gis_join_filename, "r") as read_file:
        print("Loading in list of county GISJOINs...")
        gis_joins = json.load(read_file)
        states = gis_joins["states"]
        for state_key, state_value in states.items():
            for county_key, county_value in state_value["counties"].items():
                county_gis_joins[county_value["GISJOIN"]] = []

    return county_gis_joins
