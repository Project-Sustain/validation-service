import sys
import json
from pymongo import MongoClient
from progressbar import ProgressBar, Bar, Percentage, SimpleProgress, Timer
from logging import info

# sys.path.append('./overlay')
from overlay.constants import DB_HOST, DB_PORT, DB_NAME

# Progress Bar widgets
widgets = [SimpleProgress(), Percentage(), Bar(), Timer()]


# Takes a mapping of { shard_name -> ShardMetadata } objects as input.
# Discovers which shard each GISJOIN chunk belongs to, and adds it to the list of GISJOINs for each ShardMetadata.
# Example
#   Before: ShardMetadata{ shard_name="shard7rs", ..., gis_joins=[] };
#   After:  ShardMetadata{ shard_name="shard7rs", ..., gis_joins=["G1900430", "G1800590", ...] }
# Finally, returns a mapping of { gis_join -> ShardMetadata } allowing us to do O(1) lookups to find out which shard
# a GISJOIN chunk belongs to.
def discover_gis_join_chunk_locations(shard_metadata):
    resources_dir = 'overlay/resources'
    county_gis_joins = load_gis_joins(resources_dir)
    info(f"Loaded in county GISJOIN list of size {len(county_gis_joins)}, retrieving chunk locations from MongoDB...")

    mongo_client = MongoClient(f"mongodb://{DB_HOST}:{DB_PORT}")
    db = mongo_client[DB_NAME]
    collection = db["noaa_nam"]
    gis_joins_to_shards = {}
    counter = 0
    bar = ProgressBar(maxval=len(county_gis_joins), widgets=widgets).start()
    for gis_join in county_gis_joins:
        explained_query = collection.find({"COUNTY_GISJOIN": gis_join}).explain()
        winning_plan = explained_query["queryPlanner"]["winningPlan"]
        shards = winning_plan["shards"]

        if len(shards) == 1:
            winning_shard = shards[0]
            winning_shard_name = winning_shard["shardName"]
            shard_metadata[winning_shard_name].gis_joins.append(gis_join)
            gis_joins_to_shards[gis_join] = shard_metadata[winning_shard_name]

        counter += 1
        bar.update(counter)

    bar.finish()
    return gis_joins_to_shards


# Loads all county GISJOIN values from gis_joins.json as a list.
def load_gis_joins(resources_dir):
    county_gis_joins = []
    gis_join_filename = f"{resources_dir}/gis_joins.json"
    with open(gis_join_filename, "r") as read_file:
        print("Loading in list of county GISJOINs...")
        gis_joins = json.load(read_file)
        states = gis_joins["states"]
        for state_key, state_value in states.items():
            for county_key, county_value in state_value["counties"].items():
                county_gis_joins.append(county_value["GISJOIN"])

    return county_gis_joins
