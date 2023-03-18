import sys
import json
from pymongo import MongoClient
from logging import info, error


from overlay.constants import DB_HOST, DB_PORT, username, password
from overlay.validation_pb2 import ReplicaSetMembership


class ShardMetadata:

    def __init__(self, shard_name: str, shard_servers: list, gis_join_metadata: dict):
        self.shard_name: str = shard_name                   # shard7rs
        self.shard_servers: list = shard_servers            # list(<shard_server_hostnames>)
        self.gis_join_metadata: dict = gis_join_metadata    # dict { gis_join -> count }

    def __repr__(self):
        return f"ShardMetadata: shard_name={self.shard_name}, shard_servers={self.shard_servers}, gis_joins={self.gis_joins}"

    def to_json(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)


# Discovers MongoDB cluster shards via mongo adminCommand({listShards: 1}).
# Returns a mapping of { shard_name -> ShardMetadata }
# i.e. {
#       "shard10rs": ShardMetadata{
#                                   shard_name="shard10rs",
#                                   shard_servers=["lattice-132", "lattice-133", "lattice-134"],
#                                   gis_joins=[
#                                       {"G1234567": 12345},
#                                       ...
#                                   ]
#                                  }
#       }
def discover_shards():
    info("Discovering MongoDB shards...")
    client = MongoClient(f"mongodb://{username}:{password}@{DB_HOST}:{DB_PORT}")
    shard_status = client.admin.command({"listShards": 1})
    shard_metadata = None
    if shard_status["ok"] == 1.0:
        shard_metadata = {}
        for mongo_shard in shard_status["shards"]:
            shard_name = mongo_shard["_id"]
            # 'host': 'shard2rs/lattice-108:27017,lattice-109:27017,lattice-110:27017'
            host_fields = mongo_shard["host"].split("/")[1].split(",")
            shard_servers = [host_uri.split(":")[0] for host_uri in host_fields]
            shard_metadata[shard_name] = ShardMetadata(
                shard_name,
                shard_servers,
                []
            )
            info(f"Discovered shard: {mongo_shard['host']}")

    else:
        error(f"Shard status not ok -- value {shard_status['ok']}, returning None")

    client.close()
    return shard_metadata


# Discovers and returns the replica set name and membership state of the local mongod instance
# Example return value: ("shard7rs", ReplicaSetMembership.SECONDARY)
def get_rs_member_state() -> (str, ReplicaSetMembership):
    info("Discovering MongoDB shards...")

    client = MongoClient(f"mongodb://{username}:{password}@{DB_HOST}:27017")
    rs_status = client.admin.command({"replSetGetStatus": 1})
    rs_name: str = rs_status["set"]
    rs_state: int = rs_status["myState"]  # https://www.mongodb.com/docs/manual/reference/replica-states/
    client.close()

    if rs_state == 1:
        return rs_name, ReplicaSetMembership.PRIMARY
    else:
        return rs_name, ReplicaSetMembership.SECONDARY
