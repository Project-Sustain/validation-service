from concurrent import futures
import logging
import grpc
from . import validation_pb2, validation_pb2_grpc


class Greeter(validation_pb2_grpc.GreeterServicer):

    def SayHello(self, request, context):
        return validation_pb2.HelloReply(message='Hello, %s!' % request.name)


def serve(master_hostname):
    logging.basicConfig()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    validation_pb2_grpc.add_GreeterServicer_to_server(Greeter(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()