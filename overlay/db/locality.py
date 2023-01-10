import sys
import json
import os
import socket
from pymongo import MongoClient
from progressbar import ProgressBar, Bar, Percentage, SimpleProgress, Timer
from loguru import logger

import urllib

from overlay.constants import DB_HOST, DB_PORT, DB_NAME, DB_USERNAME, DB_PASSWORD
from overlay.db.shards import ShardMetadata

# Progress Bar widgets
widgets = [SimpleProgress(), Percentage(), Bar(), Timer()]


GIS_JOIN_CHUNK_LOCATION_FILE = "overlay/resources/gis_join_chunk_locations.json"


# Finds all the GISJOINs belonging to the local mongod instance and their document counts.
# Returns a dict { gis_join -> count }
def discover_gis_joins() -> dict:
    # Connect to local mongod instance; connecting to mongos instance will find all GISJOINs in entire cluster,
    # rather than just the local shards.
    gis_join_counts: dict = {}  # { gis_join -> count }
    logger.info("Inside locality.py, just above the call to mongod")
    logger.info("Inside locality.py, password and username: ", DB_PASSWORD, DB_USERNAME)

    # client: MongoClient = MongoClient("mongodb://localhost:27017")


    client: MongoClient = MongoClient(f"mongodb://{DB_USERNAME}:{DB_PASSWORD}@localhost:27017")
    logger.info("below error")
    db = client["sustaindb"]
    coll = db["noaa_nam"]
    distinct_gis_joins: list = coll.distinct("GISJOIN")
    for gis_join in distinct_gis_joins:
        count = coll.count_documents({"GISJOIN": gis_join})
        gis_join_counts[gis_join] = count
        logger.info(f"gis_join={gis_join}, count={count}")

    client.close()
    return gis_join_counts


# Decides whether to load in cached GISJOIN locations from a saved file,
# or discover them via mongo
def get_gis_join_chunk_locations(shard_metadata: dict) -> dict:
    if gis_join_chunk_locations_file_exists():
        logger.info(f"Cached GISJOIN chunk locations file exists at {GIS_JOIN_CHUNK_LOCATION_FILE}; loading from file")
        return load_gis_join_chunk_locations(shard_metadata)
    else:
        logger.info(f"No cached GISJOIN chunk locations file found; discovering chunk locations via MongoDB queries")
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
    logger.info(f"Loaded in county GISJOIN list of size {len(county_gis_joins)}, retrieving chunk locations from MongoDB...")

    mongo_client = MongoClient(f"mongodb://{DB_USERNAME}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}")
    db = mongo_client[DB_NAME]
    collection = db["noaa_nam"]
    gis_joins_to_shards = {}
    counter = 0
    bar = ProgressBar(maxval=len(county_gis_joins), widgets=widgets).start()
    for gis_join in county_gis_joins:
        explained_query = collection.find({"GISJOIN": gis_join}).explain()
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

    logger.info(f"Saving GISJOIN chunk locations to {GIS_JOIN_CHUNK_LOCATION_FILE}")
    save_gis_join_chunk_locations(gis_joins_to_shards)

    return gis_joins_to_shards


# Discovers the number of records for each GISJOIN
def discover_gis_join_counts():
    resources_dir = 'overlay/resources'
    gis_join_counts_filename = f"{resources_dir}/gis_join_counts.json"
    if os.path.exists(gis_join_counts_filename):
        logger.info(f"Cached file {gis_join_counts_filename} already exists, skipping discovering counts")
        return

    logger.info(f"No cached {gis_join_counts_filename} file exists, discovering counts...")
    gis_join_counts = {}
    county_gis_joins = load_gis_joins(resources_dir)
    mongo_client = MongoClient(f"mongodb://{DB_USERNAME}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}")
    db = mongo_client[DB_NAME]
    collection = db["noaa_nam"]
    counter = 0
    bar = ProgressBar(maxval=len(county_gis_joins), widgets=widgets).start()
    for gis_join in county_gis_joins:
        count = collection.find({"GISJOIN": gis_join}).count()
        gis_join_counts[gis_join] = count
        counter += 1
        bar.update(counter)

    bar.finish()

    logger.info(f"Saving GISJOIN counts to {gis_join_counts_filename}")
    with open(gis_join_counts_filename, "w") as json_file:
        json.dump(gis_join_counts, json_file)
    logger.info("Success")


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
        logger.info(f"Saved {GIS_JOIN_CHUNK_LOCATION_FILE}")


def gis_join_chunk_locations_file_exists() -> bool:
    return os.path.exists(GIS_JOIN_CHUNK_LOCATION_FILE)


def get_hostname():
    return socket.gethostname()


# Returns one of ["SECONDARY", "PRIMARY", "NOT_FOUND"]
def get_replica_set_status() -> str:
    client = MongoClient(f"mongodb://{DB_USERNAME}:{DB_PASSWORD}@localhost:27017")
    repl_set_status = client.admin.command("replSetGetStatus")
    for member in repl_set_status["members"]:
        member_name = member["name"]
        member_host = member_name.split(":")[0]  # 'name': 'lattice-132:27017'
        if member_host == get_hostname():
            client.close()
            return member["stateStr"]

    client.close()
    return "NOT_FOUND"

