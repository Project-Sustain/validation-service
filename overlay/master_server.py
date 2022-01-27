import grpc
import validation_pb2
import validation_pb2_grpc
import data_locality
from logging import info, error
from concurrent import futures


class WorkerMetadata:

    def __init__(self, hostname, port):
        self.hostname = hostname
        self.port = port
        self.jobs = []

    def __repr__(self):
        return f"Worker: hostname={self.hostname}, port={self.port}, jobs={self.jobs}"


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
            for file_chunk in request_iterator:
                file_id = file_chunk.id
                file_bytes = file_chunk.data
                info(f"Length of chunk index {chunk_index}: {len(file_bytes)}")
                chunk_index += 1
                total_bytes_received += len(file_bytes)

            info(f"Finished receiving chunks, {total_bytes_received} total bytes received")
            return validation_pb2.UPLOAD_STATUS_CODE_OK

        except Exception as e:
            error(f"Failed to receive chunk index {chunk_index}: {e}")
            return validation_pb2.UPLOAD_STATUS_CODE_FAILED


def run(master_port=50051):
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
