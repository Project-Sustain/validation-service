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
  serialized_pb=b'\n\x10validation.proto\";\n\x19WorkerRegistrationRequest\x12\x10\n\x08hostname\x18\x01 \x01(\t\x12\x0c\n\x04port\x18\x02 \x01(\x05\"-\n\x1aWorkerRegistrationResponse\x12\x0f\n\x07success\x18\x01 \x01(\x08\"\xf2\x03\n\x14ValidationJobRequest\x12\n\n\x02id\x18\x01 \x01(\t\x12!\n\x0fmaster_job_mode\x18\x02 \x01(\x0e\x32\x08.JobMode\x12!\n\x0fworker_job_mode\x18\x03 \x01(\x0e\x32\x08.JobMode\x12(\n\x0fmodel_framework\x18\x04 \x01(\x0e\x32\x0f.ModelFramework\x12&\n\x0emodel_category\x18\x05 \x01(\x0e\x32\x0e.ModelCategory\x12\x12\n\nmongo_host\x18\x06 \x01(\t\x12\x12\n\nmongo_port\x18\x07 \x01(\x05\x12%\n\x0bread_config\x18\x08 \x01(\x0b\x32\x10.MongoReadConfig\x12\x10\n\x08\x64\x61tabase\x18\t \x01(\t\x12\x12\n\ncollection\x18\n \x01(\t\x12\x16\n\x0e\x66\x65\x61ture_fields\x18\x0b \x03(\t\x12\x13\n\x0blabel_field\x18\x0c \x01(\t\x12\x18\n\x10normalize_inputs\x18\r \x01(\x08\x12,\n\x11validation_budget\x18\x0e \x01(\x0b\x32\x11.ValidationBudget\x12\x19\n\x11validation_metric\x18\x0f \x01(\t\x12\x11\n\tgis_joins\x18\x10 \x03(\t\x12\x1e\n\nmodel_file\x18\x11 \x01(\x0b\x32\n.ModelFile\"f\n\x15ValidationJobResponse\x12\n\n\x02id\x18\x01 \x01(\t\x12\n\n\x02ok\x18\x02 \x01(\x08\x12\x11\n\terror_msg\x18\x03 \x01(\t\x12\"\n\x07metrics\x18\x04 \x03(\x0b\x32\x11.ValidationMetric\"D\n\x10ValidationMetric\x12\x10\n\x08gis_join\x18\x01 \x01(\t\x12\x0c\n\x04loss\x18\x02 \x01(\x01\x12\x10\n\x08\x61\x63\x63uracy\x18\x03 \x01(\x01\"9\n\tModelFile\x12\x0c\n\x04type\x18\x01 \x01(\t\x12\x10\n\x08md5_hash\x18\x02 \x01(\t\x12\x0c\n\x04\x64\x61ta\x18\x03 \x01(\x0c\"a\n\x0cUploadStatus\x12\x0f\n\x07message\x18\x01 \x01(\t\x12\x11\n\tfile_hash\x18\x02 \x01(\t\x12-\n\x12upload_status_code\x18\x03 \x01(\x0e\x32\x11.UploadStatusCode\"@\n\x0fMongoReadConfig\x12\x17\n\x0fread_preference\x18\x01 \x01(\t\x12\x14\n\x0cread_concern\x18\x02 \x01(\t\"\x8f\x01\n\x10ValidationBudget\x12 \n\x0b\x62udget_type\x18\x01 \x01(\x0e\x32\x0b.BudgetType\x12$\n\rstatic_budget\x18\x02 \x01(\x0b\x32\r.StaticBudget\x12\x33\n\x0fvariance_budget\x18\x03 \x01(\x0b\x32\x1a.IncrementalVarianceBudget\"7\n\x19IncrementalVarianceBudget\x12\x1a\n\x12initial_allocation\x18\x01 \x01(\x03\"N\n\x0cStaticBudget\x12\x13\n\x0btotal_limit\x18\x01 \x01(\x03\x12\x14\n\x0cstrata_limit\x18\x02 \x01(\x03\x12\x13\n\x0bsample_rate\x18\x03 \x01(\x02*l\n\x10UploadStatusCode\x12\x1e\n\x1aUPLOAD_STATUS_CODE_UNKNOWN\x10\x00\x12\x19\n\x15UPLOAD_STATUS_CODE_OK\x10\x01\x12\x1d\n\x19UPLOAD_STATUS_CODE_FAILED\x10\x02*@\n\nBudgetType\x12\x11\n\rSTATIC_BUDGET\x10\x00\x12\x1f\n\x1bINCREMENTAL_VARIANCE_BUDGET\x10\x01*?\n\x07JobMode\x12\x0f\n\x0bSYNCHRONOUS\x10\x00\x12\x10\n\x0c\x41SYNCHRONOUS\x10\x01\x12\x11\n\rMULTITHREADED\x10\x02*?\n\x0eModelFramework\x12\x0e\n\nTENSORFLOW\x10\x00\x12\x10\n\x0cSCIKIT_LEARN\x10\x01\x12\x0b\n\x07PYTORCH\x10\x02*\x1f\n\rModelCategory\x12\x0e\n\nREGRESSION\x10\x00\x32\x97\x02\n\x06Master\x12)\n\nUploadFile\x12\n.ModelFile\x1a\r.UploadStatus\"\x00\x12\x46\n\x13SubmitValidationJob\x12\x15.ValidationJobRequest\x1a\x16.ValidationJobResponse\"\x00\x12K\n\x0eRegisterWorker\x12\x1a.WorkerRegistrationRequest\x1a\x1b.WorkerRegistrationResponse\"\x00\x12M\n\x10\x44\x65registerWorker\x12\x1a.WorkerRegistrationRequest\x1a\x1b.WorkerRegistrationResponse\"\x00\x32O\n\x06Worker\x12\x45\n\x12\x42\x65ginValidationJob\x12\x15.ValidationJobRequest\x1a\x16.ValidationJobResponse\"\x00\x62\x06proto3'
)

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
  serialized_start=1310,
  serialized_end=1418,
)
_sym_db.RegisterEnumDescriptor(_UPLOADSTATUSCODE)

UploadStatusCode = enum_type_wrapper.EnumTypeWrapper(_UPLOADSTATUSCODE)
_BUDGETTYPE = _descriptor.EnumDescriptor(
  name='BudgetType',
  full_name='BudgetType',
  filename=None,
  file=DESCRIPTOR,
  create_key=_descriptor._internal_create_key,
  values=[
    _descriptor.EnumValueDescriptor(
      name='STATIC_BUDGET', index=0, number=0,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='INCREMENTAL_VARIANCE_BUDGET', index=1, number=1,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=1420,
  serialized_end=1484,
)
_sym_db.RegisterEnumDescriptor(_BUDGETTYPE)

BudgetType = enum_type_wrapper.EnumTypeWrapper(_BUDGETTYPE)
_JOBMODE = _descriptor.EnumDescriptor(
  name='JobMode',
  full_name='JobMode',
  filename=None,
  file=DESCRIPTOR,
  create_key=_descriptor._internal_create_key,
  values=[
    _descriptor.EnumValueDescriptor(
      name='SYNCHRONOUS', index=0, number=0,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='ASYNCHRONOUS', index=1, number=1,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='MULTITHREADED', index=2, number=2,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=1486,
  serialized_end=1549,
)
_sym_db.RegisterEnumDescriptor(_JOBMODE)

JobMode = enum_type_wrapper.EnumTypeWrapper(_JOBMODE)
_MODELFRAMEWORK = _descriptor.EnumDescriptor(
  name='ModelFramework',
  full_name='ModelFramework',
  filename=None,
  file=DESCRIPTOR,
  create_key=_descriptor._internal_create_key,
  values=[
    _descriptor.EnumValueDescriptor(
      name='TENSORFLOW', index=0, number=0,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='SCIKIT_LEARN', index=1, number=1,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='PYTORCH', index=2, number=2,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=1551,
  serialized_end=1614,
)
_sym_db.RegisterEnumDescriptor(_MODELFRAMEWORK)

ModelFramework = enum_type_wrapper.EnumTypeWrapper(_MODELFRAMEWORK)
_MODELCATEGORY = _descriptor.EnumDescriptor(
  name='ModelCategory',
  full_name='ModelCategory',
  filename=None,
  file=DESCRIPTOR,
  create_key=_descriptor._internal_create_key,
  values=[
    _descriptor.EnumValueDescriptor(
      name='REGRESSION', index=0, number=0,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=1616,
  serialized_end=1647,
)
_sym_db.RegisterEnumDescriptor(_MODELCATEGORY)

ModelCategory = enum_type_wrapper.EnumTypeWrapper(_MODELCATEGORY)
UPLOAD_STATUS_CODE_UNKNOWN = 0
UPLOAD_STATUS_CODE_OK = 1
UPLOAD_STATUS_CODE_FAILED = 2
STATIC_BUDGET = 0
INCREMENTAL_VARIANCE_BUDGET = 1
SYNCHRONOUS = 0
ASYNCHRONOUS = 1
MULTITHREADED = 2
TENSORFLOW = 0
SCIKIT_LEARN = 1
PYTORCH = 2
REGRESSION = 0



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
      name='master_job_mode', full_name='ValidationJobRequest.master_job_mode', index=1,
      number=2, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='worker_job_mode', full_name='ValidationJobRequest.worker_job_mode', index=2,
      number=3, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='model_framework', full_name='ValidationJobRequest.model_framework', index=3,
      number=4, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='model_category', full_name='ValidationJobRequest.model_category', index=4,
      number=5, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='mongo_host', full_name='ValidationJobRequest.mongo_host', index=5,
      number=6, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='mongo_port', full_name='ValidationJobRequest.mongo_port', index=6,
      number=7, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='read_config', full_name='ValidationJobRequest.read_config', index=7,
      number=8, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='database', full_name='ValidationJobRequest.database', index=8,
      number=9, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='collection', full_name='ValidationJobRequest.collection', index=9,
      number=10, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='feature_fields', full_name='ValidationJobRequest.feature_fields', index=10,
      number=11, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='label_field', full_name='ValidationJobRequest.label_field', index=11,
      number=12, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='normalize_inputs', full_name='ValidationJobRequest.normalize_inputs', index=12,
      number=13, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='validation_budget', full_name='ValidationJobRequest.validation_budget', index=13,
      number=14, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='validation_metric', full_name='ValidationJobRequest.validation_metric', index=14,
      number=15, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='gis_joins', full_name='ValidationJobRequest.gis_joins', index=15,
      number=16, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='model_file', full_name='ValidationJobRequest.model_file', index=16,
      number=17, type=11, cpp_type=10, label=1,
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
  serialized_end=627,
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
      name='id', full_name='ValidationJobResponse.id', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='ok', full_name='ValidationJobResponse.ok', index=1,
      number=2, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='error_msg', full_name='ValidationJobResponse.error_msg', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='metrics', full_name='ValidationJobResponse.metrics', index=3,
      number=4, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
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
  serialized_start=629,
  serialized_end=731,
)


_VALIDATIONMETRIC = _descriptor.Descriptor(
  name='ValidationMetric',
  full_name='ValidationMetric',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='gis_join', full_name='ValidationMetric.gis_join', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='loss', full_name='ValidationMetric.loss', index=1,
      number=2, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='accuracy', full_name='ValidationMetric.accuracy', index=2,
      number=3, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
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
  serialized_start=733,
  serialized_end=801,
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
  serialized_start=803,
  serialized_end=860,
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
  serialized_start=862,
  serialized_end=959,
)


_MONGOREADCONFIG = _descriptor.Descriptor(
  name='MongoReadConfig',
  full_name='MongoReadConfig',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='read_preference', full_name='MongoReadConfig.read_preference', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='read_concern', full_name='MongoReadConfig.read_concern', index=1,
      number=2, type=9, cpp_type=9, label=1,
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
  serialized_start=961,
  serialized_end=1025,
)


_VALIDATIONBUDGET = _descriptor.Descriptor(
  name='ValidationBudget',
  full_name='ValidationBudget',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='budget_type', full_name='ValidationBudget.budget_type', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='static_budget', full_name='ValidationBudget.static_budget', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='variance_budget', full_name='ValidationBudget.variance_budget', index=2,
      number=3, type=11, cpp_type=10, label=1,
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
  serialized_start=1028,
  serialized_end=1171,
)


_INCREMENTALVARIANCEBUDGET = _descriptor.Descriptor(
  name='IncrementalVarianceBudget',
  full_name='IncrementalVarianceBudget',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='initial_allocation', full_name='IncrementalVarianceBudget.initial_allocation', index=0,
      number=1, type=3, cpp_type=2, label=1,
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
  serialized_start=1173,
  serialized_end=1228,
)


_STATICBUDGET = _descriptor.Descriptor(
  name='StaticBudget',
  full_name='StaticBudget',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='total_limit', full_name='StaticBudget.total_limit', index=0,
      number=1, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='strata_limit', full_name='StaticBudget.strata_limit', index=1,
      number=2, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='sample_rate', full_name='StaticBudget.sample_rate', index=2,
      number=3, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
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
  serialized_start=1230,
  serialized_end=1308,
)

_VALIDATIONJOBREQUEST.fields_by_name['master_job_mode'].enum_type = _JOBMODE
_VALIDATIONJOBREQUEST.fields_by_name['worker_job_mode'].enum_type = _JOBMODE
_VALIDATIONJOBREQUEST.fields_by_name['model_framework'].enum_type = _MODELFRAMEWORK
_VALIDATIONJOBREQUEST.fields_by_name['model_category'].enum_type = _MODELCATEGORY
_VALIDATIONJOBREQUEST.fields_by_name['read_config'].message_type = _MONGOREADCONFIG
_VALIDATIONJOBREQUEST.fields_by_name['validation_budget'].message_type = _VALIDATIONBUDGET
_VALIDATIONJOBREQUEST.fields_by_name['model_file'].message_type = _MODELFILE
_VALIDATIONJOBRESPONSE.fields_by_name['metrics'].message_type = _VALIDATIONMETRIC
_UPLOADSTATUS.fields_by_name['upload_status_code'].enum_type = _UPLOADSTATUSCODE
_VALIDATIONBUDGET.fields_by_name['budget_type'].enum_type = _BUDGETTYPE
_VALIDATIONBUDGET.fields_by_name['static_budget'].message_type = _STATICBUDGET
_VALIDATIONBUDGET.fields_by_name['variance_budget'].message_type = _INCREMENTALVARIANCEBUDGET
DESCRIPTOR.message_types_by_name['WorkerRegistrationRequest'] = _WORKERREGISTRATIONREQUEST
DESCRIPTOR.message_types_by_name['WorkerRegistrationResponse'] = _WORKERREGISTRATIONRESPONSE
DESCRIPTOR.message_types_by_name['ValidationJobRequest'] = _VALIDATIONJOBREQUEST
DESCRIPTOR.message_types_by_name['ValidationJobResponse'] = _VALIDATIONJOBRESPONSE
DESCRIPTOR.message_types_by_name['ValidationMetric'] = _VALIDATIONMETRIC
DESCRIPTOR.message_types_by_name['ModelFile'] = _MODELFILE
DESCRIPTOR.message_types_by_name['UploadStatus'] = _UPLOADSTATUS
DESCRIPTOR.message_types_by_name['MongoReadConfig'] = _MONGOREADCONFIG
DESCRIPTOR.message_types_by_name['ValidationBudget'] = _VALIDATIONBUDGET
DESCRIPTOR.message_types_by_name['IncrementalVarianceBudget'] = _INCREMENTALVARIANCEBUDGET
DESCRIPTOR.message_types_by_name['StaticBudget'] = _STATICBUDGET
DESCRIPTOR.enum_types_by_name['UploadStatusCode'] = _UPLOADSTATUSCODE
DESCRIPTOR.enum_types_by_name['BudgetType'] = _BUDGETTYPE
DESCRIPTOR.enum_types_by_name['JobMode'] = _JOBMODE
DESCRIPTOR.enum_types_by_name['ModelFramework'] = _MODELFRAMEWORK
DESCRIPTOR.enum_types_by_name['ModelCategory'] = _MODELCATEGORY
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

ValidationMetric = _reflection.GeneratedProtocolMessageType('ValidationMetric', (_message.Message,), {
  'DESCRIPTOR' : _VALIDATIONMETRIC,
  '__module__' : 'validation_pb2'
  # @@protoc_insertion_point(class_scope:ValidationMetric)
  })
_sym_db.RegisterMessage(ValidationMetric)

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

MongoReadConfig = _reflection.GeneratedProtocolMessageType('MongoReadConfig', (_message.Message,), {
  'DESCRIPTOR' : _MONGOREADCONFIG,
  '__module__' : 'validation_pb2'
  # @@protoc_insertion_point(class_scope:MongoReadConfig)
  })
_sym_db.RegisterMessage(MongoReadConfig)

ValidationBudget = _reflection.GeneratedProtocolMessageType('ValidationBudget', (_message.Message,), {
  'DESCRIPTOR' : _VALIDATIONBUDGET,
  '__module__' : 'validation_pb2'
  # @@protoc_insertion_point(class_scope:ValidationBudget)
  })
_sym_db.RegisterMessage(ValidationBudget)

IncrementalVarianceBudget = _reflection.GeneratedProtocolMessageType('IncrementalVarianceBudget', (_message.Message,), {
  'DESCRIPTOR' : _INCREMENTALVARIANCEBUDGET,
  '__module__' : 'validation_pb2'
  # @@protoc_insertion_point(class_scope:IncrementalVarianceBudget)
  })
_sym_db.RegisterMessage(IncrementalVarianceBudget)

StaticBudget = _reflection.GeneratedProtocolMessageType('StaticBudget', (_message.Message,), {
  'DESCRIPTOR' : _STATICBUDGET,
  '__module__' : 'validation_pb2'
  # @@protoc_insertion_point(class_scope:StaticBudget)
  })
_sym_db.RegisterMessage(StaticBudget)



_MASTER = _descriptor.ServiceDescriptor(
  name='Master',
  full_name='Master',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_start=1650,
  serialized_end=1929,
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
  _descriptor.MethodDescriptor(
    name='DeregisterWorker',
    full_name='Master.DeregisterWorker',
    index=3,
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
  serialized_start=1931,
  serialized_end=2010,
  methods=[
  _descriptor.MethodDescriptor(
    name='BeginValidationJob',
    full_name='Worker.BeginValidationJob',
    index=0,
    containing_service=None,
    input_type=_VALIDATIONJOBREQUEST,
    output_type=_VALIDATIONJOBRESPONSE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
])
_sym_db.RegisterServiceDescriptor(_WORKER)

DESCRIPTOR.services_by_name['Worker'] = _WORKER

# @@protoc_insertion_point(module_scope)
