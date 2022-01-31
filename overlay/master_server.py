import grpc
import validation_pb2
import validation_pb2_grpc
import data_locality
import file_chunker
from logging import info, error
from concurrent import futures
import hashlib

LOCAL_TESTING = False


class WorkerMetadata:

    def __init__(self, hostname, port, gis_joins):
        self.hostname = hostname
        self.port = port
        self.jobs = []
        self.gis_joins = gis_joins

    def add_gis_join(self, gis_join):
        self.gis_joins.append(gis_join)

    def __repr__(self):
        return f"WorkerMetadata: hostname={self.hostname}, port={self.port}, jobs={self.jobs}, gis_joins={self.gis_joins}"


class Master(validation_pb2_grpc.MasterServicer):

    def __init__(self, gis_join_locations):
        super(Master, self).__init__()
        self.tracked_workers = []
        self.saved_models_path = "testing/master/saved_models"
        self.gis_join_locations = gis_join_locations

    def RegisterWorker(self, request, context):
        info(f"Received WorkerRegistrationRequest: hostname={request.hostname}, port={request.port}")
        gis_joins = []
        for gis_join, servers in self.gis_join_locations.items():
            for server in servers:
                if server == request.hostname:
                    gis_joins.append(gis_join)

        new_worker = WorkerMetadata(request.hostname, request.port, gis_joins)
        self.tracked_workers.append(new_worker)
        info(f"Added Worker: {new_worker}")
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
                    file_pointer = open(f"{self.saved_models_path}/{file_id}", "wb")

                file_bytes = file_chunk.data
                info(f"Length of chunk index {chunk_index}: {len(file_bytes)}")
                chunk_index += 1
                total_bytes_received += len(file_bytes)
                file_pointer.write(file_bytes)

            file_pointer.close()
            info(f"Finished receiving chunks, {total_bytes_received} total bytes received")

            # Get file hash
            if file_id:
                with open(f"{self.saved_models_path}/{file_id}", "rb") as f:
                    hasher = hashlib.md5()
                    buf = f.read()
                    hasher.update(buf)
                info(f"Uploaded file hash: {hasher.hexdigest()}")

                # Upload file to all workers
                for worker in self.tracked_workers:
                    with open(f"{self.saved_models_path}/{file_id}", "rb") as f:
                        with grpc.insecure_channel(f"{worker.hostname}:{worker.port}") as channel:
                            stub = validation_pb2_grpc.WorkerStub(channel)
                            upload_response = stub.UploadFile(file_chunker.chunk_file(f))
                            if upload_response == validation_pb2.UPLOAD_STATUS_CODE_FAILED:
                                raise ValueError(f"Failed to upload file {file_id} to worker {worker.hostname}")
                    info(f"Successfully uploaded file to worker {worker.hostname}")

                # Success
                return validation_pb2.UploadStatus(
                    message="Success",
                    file_hash=hasher.hexdigest(),
                    upload_status_code=validation_pb2.UPLOAD_STATUS_CODE_OK
                )

        except ValueError as e:
            error(f"{e}")
        except Exception as e:
            error(f"Failed to receive chunk index {chunk_index}: {e}")

        # Failure, hopefully we don't make it here
        return validation_pb2.UploadStatus(
            message="Failed",
            file_hash="None",
            upload_status_code=validation_pb2.UPLOAD_STATUS_CODE_FAILED
        )

    def SubmitValidationJob(self, request, context):
        info(f"SubmitValidationJob Request: {request}")
        for worker in self.tracked_workers:
            info(f"Submitting validation job to {worker} for GISJOINs {worker.gis_joins}")

            break  # TODO: Remove
        return validation_pb2.ValidationJobResponse(message="OK")


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
