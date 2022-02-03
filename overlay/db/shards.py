import sys
from pymongo import MongoClient
from logging import info, error

from .. import DB_HOST, DB_PORT, DB_NAME


class ShardMetadata:

    def __init__(self, shard_name, shard_servers):
        self.shard_name = shard_name
        self.shard_servers = shard_servers
        self.gis_joins = []

    def __repr__(self):
        return f"ShardMetadata: shard_name={self.shard_name}, shard_servers={self.shard_servers}, {len(self.gis_joins)} gis_joins"


# Discovers MongoDB cluster shards via mongo adminCommand({listShards: 1}).
# Returns a mapping of { shard_name -> ShardMetadata }
# i.e. {
#       "shard10rs": ShardMetadata{
#                                   shard_name="shard10rs",
#                                   shard_servers=["lattice-132", "lattice-133", "lattice-134"],
#                                   gis_joins=[]
#                                  }
#       }
def discover_shards():
    info("Discovering MongoDB shards...")
    client = MongoClient(f"mongodb://{DB_HOST}:{DB_PORT}")
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
                shard_servers
            )
            info(f"Discovered shard: {mongo_shard['host']}")

    else:
        error(f"Shard status not ok -- value {shard_status['ok']}, returning None")

    client.close()
    return shard_metadata
