# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import validation_pb2 as validation__pb2


class MasterStub(object):
    """Master service definition
    """

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.UploadFile = channel.stream_unary(
                '/Master/UploadFile',
                request_serializer=validation__pb2.FileChunk.SerializeToString,
                response_deserializer=validation__pb2.UploadStatus.FromString,
                )
        self.RegisterWorker = channel.unary_unary(
                '/Master/RegisterWorker',
                request_serializer=validation__pb2.WorkerRegistrationRequest.SerializeToString,
                response_deserializer=validation__pb2.WorkerRegistrationResponse.FromString,
                )


class MasterServicer(object):
    """Master service definition
    """

    def UploadFile(self, request_iterator, context):
        """Allows streamed uploading of a .zip model to the Master
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def RegisterWorker(self, request, context):
        """Registers a Worker to track
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_MasterServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'UploadFile': grpc.stream_unary_rpc_method_handler(
                    servicer.UploadFile,
                    request_deserializer=validation__pb2.FileChunk.FromString,
                    response_serializer=validation__pb2.UploadStatus.SerializeToString,
            ),
            'RegisterWorker': grpc.unary_unary_rpc_method_handler(
                    servicer.RegisterWorker,
                    request_deserializer=validation__pb2.WorkerRegistrationRequest.FromString,
                    response_serializer=validation__pb2.WorkerRegistrationResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'Master', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class Master(object):
    """Master service definition
    """

    @staticmethod
    def UploadFile(request_iterator,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.stream_unary(request_iterator, target, '/Master/UploadFile',
            validation__pb2.FileChunk.SerializeToString,
            validation__pb2.UploadStatus.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def RegisterWorker(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/Master/RegisterWorker',
            validation__pb2.WorkerRegistrationRequest.SerializeToString,
            validation__pb2.WorkerRegistrationResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)


class WorkerStub(object):
    """Worker service definition
    """

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.UploadFile = channel.stream_unary(
                '/Worker/UploadFile',
                request_serializer=validation__pb2.FileChunk.SerializeToString,
                response_deserializer=validation__pb2.UploadStatus.FromString,
                )
        self.BeginValidationJob = channel.unary_unary(
                '/Worker/BeginValidationJob',
                request_serializer=validation__pb2.WorkerJobRequest.SerializeToString,
                response_deserializer=validation__pb2.WorkerRegistrationResponse.FromString,
                )


class WorkerServicer(object):
    """Worker service definition
    """

    def UploadFile(self, request_iterator, context):
        """Allows streamed uploading of a .zip model to the Worker
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def BeginValidationJob(self, request, context):
        """Registers a Worker to track via heartbeats
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_WorkerServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'UploadFile': grpc.stream_unary_rpc_method_handler(
                    servicer.UploadFile,
                    request_deserializer=validation__pb2.FileChunk.FromString,
                    response_serializer=validation__pb2.UploadStatus.SerializeToString,
            ),
            'BeginValidationJob': grpc.unary_unary_rpc_method_handler(
                    servicer.BeginValidationJob,
                    request_deserializer=validation__pb2.WorkerJobRequest.FromString,
                    response_serializer=validation__pb2.WorkerRegistrationResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'Worker', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class Worker(object):
    """Worker service definition
    """

    @staticmethod
    def UploadFile(request_iterator,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.stream_unary(request_iterator, target, '/Worker/UploadFile',
            validation__pb2.FileChunk.SerializeToString,
            validation__pb2.UploadStatus.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def BeginValidationJob(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/Worker/BeginValidationJob',
            validation__pb2.WorkerJobRequest.SerializeToString,
            validation__pb2.WorkerRegistrationResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)