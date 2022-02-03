# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: validation.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='validation.proto',
  package='',
  syntax='proto3',
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\x10validation.proto\";\n\x19WorkerRegistrationRequest\x12\x10\n\x08hostname\x18\x01 \x01(\t\x12\x0c\n\x04port\x18\x02 \x01(\x05\"-\n\x1aWorkerRegistrationResponse\x12\x0f\n\x07success\x18\x01 \x01(\x08\"\x87\x02\n\x14ValidationJobRequest\x12\n\n\x02id\x18\x01 \x01(\t\x12\x17\n\x0fmodel_framework\x18\x02 \x01(\t\x12\x12\n\nmodel_type\x18\x03 \x01(\t\x12\x10\n\x08\x64\x61tabase\x18\x04 \x01(\t\x12\x12\n\ncollection\x18\x05 \x01(\t\x12\x15\n\rspatial_field\x18\x06 \x01(\t\x12\x13\n\x0blabel_field\x18\x07 \x01(\t\x12\x19\n\x11validation_metric\x18\x08 \x01(\t\x12\x16\n\x0e\x66\x65\x61ture_fields\x18\t \x03(\t\x12\x11\n\tgis_joins\x18\n \x03(\t\x12\x1e\n\nmodel_file\x18\x0b \x01(\x0b\x32\n.ModelFile\"(\n\x15ValidationJobResponse\x12\x0f\n\x07message\x18\x01 \x01(\t\"l\n\x11WorkerJobResponse\x12\n\n\x02id\x18\x01 \x01(\t\x12\x15\n\rerror_message\x18\x02 \x01(\t\x12\x34\n\x16worker_job_status_code\x18\x03 \x01(\x0e\x32\x14.WorkerJobStatusCode\"9\n\tModelFile\x12\x0c\n\x04type\x18\x01 \x01(\t\x12\x10\n\x08md5_hash\x18\x02 \x01(\t\x12\x0c\n\x04\x64\x61ta\x18\x03 \x01(\x0c\"a\n\x0cUploadStatus\x12\x0f\n\x07message\x18\x01 \x01(\t\x12\x11\n\tfile_hash\x18\x02 \x01(\t\x12-\n\x12upload_status_code\x18\x03 \x01(\x0e\x32\x11.UploadStatusCode*\x9d\x01\n\x13WorkerJobStatusCode\x12\"\n\x1eWORKER_JOB_STATUS_CODE_UNKNOWN\x10\x00\x12\x1d\n\x19WORKER_JOB_STATUS_CODE_OK\x10\x01\x12 \n\x1cWORKER_JOB_STATUS_CODE_ERROR\x10\x02\x12!\n\x1dWORKER_JOB_STATUS_CODE_FAILED\x10\x03*l\n\x10UploadStatusCode\x12\x1e\n\x1aUPLOAD_STATUS_CODE_UNKNOWN\x10\x00\x12\x19\n\x15UPLOAD_STATUS_CODE_OK\x10\x01\x12\x1d\n\x19UPLOAD_STATUS_CODE_FAILED\x10\x02\x32\xc8\x01\n\x06Master\x12)\n\nUploadFile\x12\n.ModelFile\x1a\r.UploadStatus\"\x00\x12\x46\n\x13SubmitValidationJob\x12\x15.ValidationJobRequest\x1a\x16.ValidationJobResponse\"\x00\x12K\n\x0eRegisterWorker\x12\x1a.WorkerRegistrationRequest\x1a\x1b.WorkerRegistrationResponse\"\x00\x32v\n\x06Worker\x12)\n\nUploadFile\x12\n.ModelFile\x1a\r.UploadStatus\"\x00\x12\x41\n\x12\x42\x65ginValidationJob\x12\x15.ValidationJobRequest\x1a\x12.WorkerJobResponse\"\x00\x62\x06proto3'
)

_WORKERJOBSTATUSCODE = _descriptor.EnumDescriptor(
  name='WorkerJobStatusCode',
  full_name='WorkerJobStatusCode',
  filename=None,
  file=DESCRIPTOR,
  create_key=_descriptor._internal_create_key,
  values=[
    _descriptor.EnumValueDescriptor(
      name='WORKER_JOB_STATUS_CODE_UNKNOWN', index=0, number=0,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='WORKER_JOB_STATUS_CODE_OK', index=1, number=1,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='WORKER_JOB_STATUS_CODE_ERROR', index=2, number=2,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='WORKER_JOB_STATUS_CODE_FAILED', index=3, number=3,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=705,
  serialized_end=862,
)
_sym_db.RegisterEnumDescriptor(_WORKERJOBSTATUSCODE)

WorkerJobStatusCode = enum_type_wrapper.EnumTypeWrapper(_WORKERJOBSTATUSCODE)
_UPLOADSTATUSCODE = _descriptor.EnumDescriptor(
  name='UploadStatusCode',
  full_name='UploadStatusCode',
  filename=None,
  file=DESCRIPTOR,
  create_key=_descriptor._internal_create_key,
  values=[
    _descriptor.EnumValueDescriptor(
      name='UPLOAD_STATUS_CODE_UNKNOWN', index=0, number=0,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='UPLOAD_STATUS_CODE_OK', index=1, number=1,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='UPLOAD_STATUS_CODE_FAILED', index=2, number=2,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=864,
  serialized_end=972,
)
_sym_db.RegisterEnumDescriptor(_UPLOADSTATUSCODE)

UploadStatusCode = enum_type_wrapper.EnumTypeWrapper(_UPLOADSTATUSCODE)
WORKER_JOB_STATUS_CODE_UNKNOWN = 0
WORKER_JOB_STATUS_CODE_OK = 1
WORKER_JOB_STATUS_CODE_ERROR = 2
WORKER_JOB_STATUS_CODE_FAILED = 3
UPLOAD_STATUS_CODE_UNKNOWN = 0
UPLOAD_STATUS_CODE_OK = 1
UPLOAD_STATUS_CODE_FAILED = 2



_WORKERREGISTRATIONREQUEST = _descriptor.Descriptor(
  name='WorkerRegistrationRequest',
  full_name='WorkerRegistrationRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='hostname', full_name='WorkerRegistrationRequest.hostname', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='port', full_name='WorkerRegistrationRequest.port', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=20,
  serialized_end=79,
)


_WORKERREGISTRATIONRESPONSE = _descriptor.Descriptor(
  name='WorkerRegistrationResponse',
  full_name='WorkerRegistrationResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='success', full_name='WorkerRegistrationResponse.success', index=0,
      number=1, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=81,
  serialized_end=126,
)


_VALIDATIONJOBREQUEST = _descriptor.Descriptor(
  name='ValidationJobRequest',
  full_name='ValidationJobRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='id', full_name='ValidationJobRequest.id', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='model_framework', full_name='ValidationJobRequest.model_framework', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='model_type', full_name='ValidationJobRequest.model_type', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='database', full_name='ValidationJobRequest.database', index=3,
      number=4, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='collection', full_name='ValidationJobRequest.collection', index=4,
      number=5, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='spatial_field', full_name='ValidationJobRequest.spatial_field', index=5,
      number=6, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='label_field', full_name='ValidationJobRequest.label_field', index=6,
      number=7, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='validation_metric', full_name='ValidationJobRequest.validation_metric', index=7,
      number=8, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='feature_fields', full_name='ValidationJobRequest.feature_fields', index=8,
      number=9, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='gis_joins', full_name='ValidationJobRequest.gis_joins', index=9,
      number=10, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='model_file', full_name='ValidationJobRequest.model_file', index=10,
      number=11, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=129,
  serialized_end=392,
)


_VALIDATIONJOBRESPONSE = _descriptor.Descriptor(
  name='ValidationJobResponse',
  full_name='ValidationJobResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='message', full_name='ValidationJobResponse.message', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=394,
  serialized_end=434,
)


_WORKERJOBRESPONSE = _descriptor.Descriptor(
  name='WorkerJobResponse',
  full_name='WorkerJobResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='id', full_name='WorkerJobResponse.id', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='error_message', full_name='WorkerJobResponse.error_message', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='worker_job_status_code', full_name='WorkerJobResponse.worker_job_status_code', index=2,
      number=3, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=436,
  serialized_end=544,
)


_MODELFILE = _descriptor.Descriptor(
  name='ModelFile',
  full_name='ModelFile',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='type', full_name='ModelFile.type', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='md5_hash', full_name='ModelFile.md5_hash', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='data', full_name='ModelFile.data', index=2,
      number=3, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=b"",
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=546,
  serialized_end=603,
)


_UPLOADSTATUS = _descriptor.Descriptor(
  name='UploadStatus',
  full_name='UploadStatus',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='message', full_name='UploadStatus.message', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='file_hash', full_name='UploadStatus.file_hash', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='upload_status_code', full_name='UploadStatus.upload_status_code', index=2,
      number=3, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=605,
  serialized_end=702,
)

_VALIDATIONJOBREQUEST.fields_by_name['model_file'].message_type = _MODELFILE
_WORKERJOBRESPONSE.fields_by_name['worker_job_status_code'].enum_type = _WORKERJOBSTATUSCODE
_UPLOADSTATUS.fields_by_name['upload_status_code'].enum_type = _UPLOADSTATUSCODE
DESCRIPTOR.message_types_by_name['WorkerRegistrationRequest'] = _WORKERREGISTRATIONREQUEST
DESCRIPTOR.message_types_by_name['WorkerRegistrationResponse'] = _WORKERREGISTRATIONRESPONSE
DESCRIPTOR.message_types_by_name['ValidationJobRequest'] = _VALIDATIONJOBREQUEST
DESCRIPTOR.message_types_by_name['ValidationJobResponse'] = _VALIDATIONJOBRESPONSE
DESCRIPTOR.message_types_by_name['WorkerJobResponse'] = _WORKERJOBRESPONSE
DESCRIPTOR.message_types_by_name['ModelFile'] = _MODELFILE
DESCRIPTOR.message_types_by_name['UploadStatus'] = _UPLOADSTATUS
DESCRIPTOR.enum_types_by_name['WorkerJobStatusCode'] = _WORKERJOBSTATUSCODE
DESCRIPTOR.enum_types_by_name['UploadStatusCode'] = _UPLOADSTATUSCODE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

WorkerRegistrationRequest = _reflection.GeneratedProtocolMessageType('WorkerRegistrationRequest', (_message.Message,), {
  'DESCRIPTOR' : _WORKERREGISTRATIONREQUEST,
  '__module__' : 'validation_pb2'
  # @@protoc_insertion_point(class_scope:WorkerRegistrationRequest)
  })
_sym_db.RegisterMessage(WorkerRegistrationRequest)

WorkerRegistrationResponse = _reflection.GeneratedProtocolMessageType('WorkerRegistrationResponse', (_message.Message,), {
  'DESCRIPTOR' : _WORKERREGISTRATIONRESPONSE,
  '__module__' : 'validation_pb2'
  # @@protoc_insertion_point(class_scope:WorkerRegistrationResponse)
  })
_sym_db.RegisterMessage(WorkerRegistrationResponse)

ValidationJobRequest = _reflection.GeneratedProtocolMessageType('ValidationJobRequest', (_message.Message,), {
  'DESCRIPTOR' : _VALIDATIONJOBREQUEST,
  '__module__' : 'validation_pb2'
  # @@protoc_insertion_point(class_scope:ValidationJobRequest)
  })
_sym_db.RegisterMessage(ValidationJobRequest)

ValidationJobResponse = _reflection.GeneratedProtocolMessageType('ValidationJobResponse', (_message.Message,), {
  'DESCRIPTOR' : _VALIDATIONJOBRESPONSE,
  '__module__' : 'validation_pb2'
  # @@protoc_insertion_point(class_scope:ValidationJobResponse)
  })
_sym_db.RegisterMessage(ValidationJobResponse)

WorkerJobResponse = _reflection.GeneratedProtocolMessageType('WorkerJobResponse', (_message.Message,), {
  'DESCRIPTOR' : _WORKERJOBRESPONSE,
  '__module__' : 'validation_pb2'
  # @@protoc_insertion_point(class_scope:WorkerJobResponse)
  })
_sym_db.RegisterMessage(WorkerJobResponse)

ModelFile = _reflection.GeneratedProtocolMessageType('ModelFile', (_message.Message,), {
  'DESCRIPTOR' : _MODELFILE,
  '__module__' : 'validation_pb2'
  # @@protoc_insertion_point(class_scope:ModelFile)
  })
_sym_db.RegisterMessage(ModelFile)

UploadStatus = _reflection.GeneratedProtocolMessageType('UploadStatus', (_message.Message,), {
  'DESCRIPTOR' : _UPLOADSTATUS,
  '__module__' : 'validation_pb2'
  # @@protoc_insertion_point(class_scope:UploadStatus)
  })
_sym_db.RegisterMessage(UploadStatus)



_MASTER = _descriptor.ServiceDescriptor(
  name='Master',
  full_name='Master',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_start=975,
  serialized_end=1175,
  methods=[
  _descriptor.MethodDescriptor(
    name='UploadFile',
    full_name='Master.UploadFile',
    index=0,
    containing_service=None,
    input_type=_MODELFILE,
    output_type=_UPLOADSTATUS,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='SubmitValidationJob',
    full_name='Master.SubmitValidationJob',
    index=1,
    containing_service=None,
    input_type=_VALIDATIONJOBREQUEST,
    output_type=_VALIDATIONJOBRESPONSE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='RegisterWorker',
    full_name='Master.RegisterWorker',
    index=2,
    containing_service=None,
    input_type=_WORKERREGISTRATIONREQUEST,
    output_type=_WORKERREGISTRATIONRESPONSE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
])
_sym_db.RegisterServiceDescriptor(_MASTER)

DESCRIPTOR.services_by_name['Master'] = _MASTER


_WORKER = _descriptor.ServiceDescriptor(
  name='Worker',
  full_name='Worker',
  file=DESCRIPTOR,
  index=1,
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_start=1177,
  serialized_end=1295,
  methods=[
  _descriptor.MethodDescriptor(
    name='UploadFile',
    full_name='Worker.UploadFile',
    index=0,
    containing_service=None,
    input_type=_MODELFILE,
    output_type=_UPLOADSTATUS,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='BeginValidationJob',
    full_name='Worker.BeginValidationJob',
    index=1,
    containing_service=None,
    input_type=_VALIDATIONJOBREQUEST,
    output_type=_WORKERJOBRESPONSE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
])
_sym_db.RegisterServiceDescriptor(_WORKER)

DESCRIPTOR.services_by_name['Worker'] = _WORKER

# @@protoc_insertion_point(module_scope)
