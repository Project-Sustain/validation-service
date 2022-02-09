import sys
import json
import os
from pymongo import MongoClient
from progressbar import ProgressBar, Bar, Percentage, SimpleProgress, Timer
from logging import info

from overlay.constants import DB_HOST, DB_PORT, DB_NAME
from overlay.db.shards import ShardMetadata

# Progress Bar widgets
widgets = [SimpleProgress(), Percentage(), Bar(), Timer()]


GIS_JOIN_CHUNK_LOCATION_FILE = "overlay/resources/gis_join_chunk_locations.json"


# Decides whether to load in cached GISJOIN locations from a saved file,
# or discover them via mongo
def get_gis_join_chunk_locations(shard_metadata: dict) -> dict:
    if gis_join_chunk_locations_file_exists():
        info(f"Cached GISJOIN chunk locations file exists at {GIS_JOIN_CHUNK_LOCATION_FILE}; loading from file")
        return load_gis_join_chunk_locations(shard_metadata)
    else:
        info(f"No cached GISJOIN chunk locations file found; discovering chunk locations via MongoDB queries")
        return discover_gis_join_chunk_locations(shard_metadata)


# Takes a mapping of { shard_name -> ShardMetadata } objects as input.
# Discovers which shard each GISJOIN chunk belongs to, and adds it to the list of GISJOINs for each ShardMetadata.
# Example
#   Before: ShardMetadata{ shard_name="shard7rs", ..., gis_joins=[] };
#   After:  ShardMetadata{ shard_name="shard7rs", ..., gis_joins=["G1900430", "G1800590", ...] }
# Finally, returns a mapping of { gis_join -> ShardMetadata } allowing us to do O(1) lookups to find out which shard
# a GISJOIN chunk belongs to.
def discover_gis_join_chunk_locations(shard_metadata: dict) -> dict:
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

    info(f"Saving GISJOIN chunk locations to {GIS_JOIN_CHUNK_LOCATION_FILE}")
    save_gis_join_chunk_locations(gis_joins_to_shards)

    return gis_joins_to_shards


# Loads all county GISJOIN values from gis_joins.json as a list.
def load_gis_joins(resources_dir: str):
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


def load_gis_join_chunk_locations(shard_metadata: dict) -> dict:
    raw_dictionary: dict = {}
    gis_join_chunk_locations: dict = {}
    with open(GIS_JOIN_CHUNK_LOCATION_FILE, "r") as f:
        raw_dictionary = json.load(f)

    for gis_join_key, value in raw_dictionary.items():
        shard_metadata_ref: ShardMetadata = shard_metadata[value["shard_name"]]
        shard_metadata_ref.gis_joins.append(gis_join_key)
        gis_join_chunk_locations[gis_join_key] = shard_metadata[value["shard_name"]]

    return gis_join_chunk_locations


def save_gis_join_chunk_locations(shard_metadata: dict) -> None:
    json_str = "{\n"
    last_entry: int = len(shard_metadata.keys()) - 1
    current_entry = 0
    for key, value in shard_metadata.items():
        json_str += f"\"{key}\": {value.to_json()}"
        json_str += ",\n" if current_entry < last_entry else "\n"
        current_entry += 1

    json_str += "}"
    with open(GIS_JOIN_CHUNK_LOCATION_FILE, "w") as f:
        f.write(json_str)
        info(f"Saved {GIS_JOIN_CHUNK_LOCATION_FILE}")


def gis_join_chunk_locations_file_exists() -> bool:
    return os.path.exists(GIS_JOIN_CHUNK_LOCATION_FILE)
