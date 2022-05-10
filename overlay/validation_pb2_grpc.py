# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

from overlay import validation_pb2 as validation__pb2


class MasterStub(object):
    """Master service definition
    """

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.UploadFile = channel.unary_unary(
                '/Master/UploadFile',
                request_serializer=validation__pb2.ModelFile.SerializeToString,
                response_deserializer=validation__pb2.UploadStatus.FromString,
                )
        self.SubmitValidationJob = channel.unary_stream(
                '/Master/SubmitValidationJob',
                request_serializer=validation__pb2.ValidationJobRequest.SerializeToString,
                response_deserializer=validation__pb2.ValidationJobResponse.FromString,
                )
        self.SubmitExperiment = channel.unary_unary(
                '/Master/SubmitExperiment',
                request_serializer=validation__pb2.ValidationJobRequest.SerializeToString,
                response_deserializer=validation__pb2.ExperimentResponse.FromString,
                )
        self.RegisterWorker = channel.unary_unary(
                '/Master/RegisterWorker',
                request_serializer=validation__pb2.WorkerRegistrationRequest.SerializeToString,
                response_deserializer=validation__pb2.WorkerRegistrationResponse.FromString,
                )
        self.DeregisterWorker = channel.unary_unary(
                '/Master/DeregisterWorker',
                request_serializer=validation__pb2.WorkerRegistrationRequest.SerializeToString,
                response_deserializer=validation__pb2.WorkerRegistrationResponse.FromString,
                )


class MasterServicer(object):
    """Master service definition
    """

    def UploadFile(self, request, context):
        """Allows streamed uploading of a .zip model to the Master
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def SubmitValidationJob(self, request, context):
        """Submits a validation job to the cluster
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def SubmitExperiment(self, request, context):
        """Submits a validation job experiment to the cluster
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

    def DeregisterWorker(self, request, context):
        """De-registers a Worker from tracking
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_MasterServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'UploadFile': grpc.unary_unary_rpc_method_handler(
                    servicer.UploadFile,
                    request_deserializer=validation__pb2.ModelFile.FromString,
                    response_serializer=validation__pb2.UploadStatus.SerializeToString,
            ),
            'SubmitValidationJob': grpc.unary_stream_rpc_method_handler(
                    servicer.SubmitValidationJob,
                    request_deserializer=validation__pb2.ValidationJobRequest.FromString,
                    response_serializer=validation__pb2.ValidationJobResponse.SerializeToString,
            ),
            'SubmitExperiment': grpc.unary_unary_rpc_method_handler(
                    servicer.SubmitExperiment,
                    request_deserializer=validation__pb2.ValidationJobRequest.FromString,
                    response_serializer=validation__pb2.ExperimentResponse.SerializeToString,
            ),
            'RegisterWorker': grpc.unary_unary_rpc_method_handler(
                    servicer.RegisterWorker,
                    request_deserializer=validation__pb2.WorkerRegistrationRequest.FromString,
                    response_serializer=validation__pb2.WorkerRegistrationResponse.SerializeToString,
            ),
            'DeregisterWorker': grpc.unary_unary_rpc_method_handler(
                    servicer.DeregisterWorker,
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
    def UploadFile(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/Master/UploadFile',
            validation__pb2.ModelFile.SerializeToString,
            validation__pb2.UploadStatus.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def SubmitValidationJob(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_stream(request, target, '/Master/SubmitValidationJob',
            validation__pb2.ValidationJobRequest.SerializeToString,
            validation__pb2.ValidationJobResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def SubmitExperiment(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/Master/SubmitExperiment',
            validation__pb2.ValidationJobRequest.SerializeToString,
            validation__pb2.ExperimentResponse.FromString,
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

    @staticmethod
    def DeregisterWorker(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/Master/DeregisterWorker',
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
        self.BeginValidationJob = channel.unary_stream(
                '/Worker/BeginValidationJob',
                request_serializer=validation__pb2.ValidationJobRequest.SerializeToString,
                response_deserializer=validation__pb2.Metric.FromString,
                )
        self.DebugBeginValidationJob = channel.unary_stream(
                '/Worker/DebugBeginValidationJob',
                request_serializer=validation__pb2.ValidationJobRequest.SerializeToString,
                response_deserializer=validation__pb2.Metric.FromString,
                )


class WorkerServicer(object):
    """Worker service definition
    """

    def BeginValidationJob(self, request, context):
        """Registers a Worker to track via heartbeats
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def DebugBeginValidationJob(self, request, context):
        """For debugging
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_WorkerServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'BeginValidationJob': grpc.unary_stream_rpc_method_handler(
                    servicer.BeginValidationJob,
                    request_deserializer=validation__pb2.ValidationJobRequest.FromString,
                    response_serializer=validation__pb2.Metric.SerializeToString,
            ),
            'DebugBeginValidationJob': grpc.unary_stream_rpc_method_handler(
                    servicer.DebugBeginValidationJob,
                    request_deserializer=validation__pb2.ValidationJobRequest.FromString,
                    response_serializer=validation__pb2.Metric.SerializeToString,
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
        return grpc.experimental.unary_stream(request, target, '/Worker/BeginValidationJob',
            validation__pb2.ValidationJobRequest.SerializeToString,
            validation__pb2.Metric.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def DebugBeginValidationJob(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_stream(request, target, '/Worker/DebugBeginValidationJob',
            validation__pb2.ValidationJobRequest.SerializeToString,
            validation__pb2.Metric.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
