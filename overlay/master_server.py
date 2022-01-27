import grpc
import validation_pb2
import validation_pb2_grpc
import data_locality
from logging import info, error
from concurrent import futures
import hashlib

LOCAL_TESTING = True
SAVE_DIR = "testing/master/saved_models"


class WorkerMetadata:

    def __init__(self, hostname, port):
        self.hostname = hostname
        self.port = port
        self.jobs = []
        self.gis_joins = []

    def add_gis_join(self, gis_join):
        self.gis_joins.append(gis_join)

    def __repr__(self):
        return f"Worker: hostname={self.hostname}, port={self.port}, jobs={self.jobs}, gis_joins={self.gis_joins}"


class Master(validation_pb2_grpc.MasterServicer):

    def __init__(self, gis_join_locations):
        super(Master, self).__init__()
        self.tracked_workers = []
        self.saved_models_path = "testing/master/saved_models"
        self.gis_join_locations = gis_join_locations

    def RegisterWorker(self, request, context):
        info(f"Received WorkerRegistrationRequest: hostname={request.hostname}, port={request.port}")
        self.tracked_workers.append(WorkerMetadata(request.hostname, request.port))
        return validation_pb2.WorkerRegistrationResponse(success=True)

    def UploadFile(self, request_iterator, context):
        info(f"Received UploadFile stream request, processing chunks...")
        total_bytes_received = 0
        chunk_index = 0

        try:
            file_pointer = None
            file_id = None
            for file_chunk in request_iterator:
                file_id = file_chunk.id
                if not file_pointer:
                    file_pointer = open(f"{SAVE_DIR}/{file_id}", "wb")

                file_bytes = file_chunk.data
                info(f"Length of chunk index {chunk_index}: {len(file_bytes)}")
                chunk_index += 1
                total_bytes_received += len(file_bytes)
                file_pointer.write(file_bytes)

            file_pointer.close()
            info(f"Finished receiving chunks, {total_bytes_received} total bytes received")

            # Get file hash
            if file_id:
                with open(f"{SAVE_DIR}/{file_id}", "rb") as f:
                    hasher = hashlib.md5()
                    buf = f.read()
                    hasher.update(buf)
                info(f"Uploaded file hash: {hasher.hexdigest()}")

                # Success
                return validation_pb2.UploadStatus(
                    message="Success",
                    file_hash=hasher.hexdigest(),
                    upload_status_code=validation_pb2.UPLOAD_STATUS_CODE_OK
                )

        except Exception as e:
            error(f"Failed to receive chunk index {chunk_index}: {e}")

        # Failure, hopefully we don't make it here
        return validation_pb2.UploadStatus(
            message="Failed",
            file_hash="None",
            upload_status_code=validation_pb2.UPLOAD_STATUS_CODE_FAILED
        )


def run(master_port=50051):
    if LOCAL_TESTING:
        gis_join_locations = {}
    else:
        gis_join_locations = data_locality.get_gis_join_locations()

    # Initialize server and master
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    master = Master(gis_join_locations)
    validation_pb2_grpc.add_MasterServicer_to_server(master, server)

    # Start the server
    info(f"Starting master server on port {master_port}")
    server.add_insecure_port(f"[::]:{master_port}")
    server.start()
    server.wait_for_termination()
