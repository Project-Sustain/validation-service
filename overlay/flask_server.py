import json
import os
import grpc
import hashlib
import jsonschema
import time
from jsonschema import validate
from flask import Flask, request, jsonify, stream_with_context
from http import HTTPStatus
from pprint import pprint

from loguru import logger
from logtail import LogtailHandler

from google.protobuf.json_format import MessageToJson, Parse, MessageToDict
from werkzeug.datastructures import FileStorage
import urllib

from overlay import validation_pb2_grpc
from overlay.validation_pb2 import ValidationJobRequest, ValidationJobResponse, ModelFileType, ExperimentResponse, \
    ResponseMetric, ModelFramework, ModelCategory, LossFunction, SpatialCoverage

UPLOAD_DIR = "./uploads"
ALLOWED_EXTENSIONS = {"zip", "pt", "pkl", "h5"}

app = Flask(__name__)
app.config["UPLOAD_DIR"] = UPLOAD_DIR

# Logtail - Validation Service - Flask Server
logtail_handler = LogtailHandler(source_token="c8MjVJuDeG5TCkLtRiCMx8ev")
logger.add(logtail_handler)


# Main entrypoint
def run(master_hostname="localhost", master_port=50051, flask_port=5000):
    username = os.environ.get('ROOT_MONGO_USER')
    password = os.environ.get('ROOT_MONGO_PASS')
    print("Testing environment variables: ", username, password)
    print("Running flask server...")
    app.config["MASTER_HOSTNAME"] = master_hostname
    app.config["MASTER_PORT"] = master_port
    app.run(host="0.0.0.0", port=flask_port)  # Entrypoint


def allowed_file(filename) -> bool:
    return "." in filename and \
        filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def file_type(filename) -> ModelFileType:
    extension = filename.rsplit(".", 1)[1].lower()
    logger.info(f"Model file extension: {extension}")
    if extension == "pt":
        return ModelFileType.PYTORCH_TORCHSCRIPT
    elif extension == "zip":
        return ModelFileType.TENSORFLOW_SAVED_MODEL_ZIP
    elif extension == "h5":
        return ModelFileType.TENSORFLOW_HDF5
    elif extension == "pkl":
        return ModelFileType.SCIKIT_LEARN_PICKLE
    else:
        return ModelFileType.UNKNOWN_MODEL_FILE_TYPE


@app.route("/validation_service", methods=["GET"])
def default_route():
    return "Welcome to the Sustain Validation Service! Query /validation_service/schema with HTTP GET for a request " \
           "schema. "


@app.route("/validation_service/schema", methods=["GET"])
def get_schema():
    logger.info(f"Received GET request for ")
    with open("./resources/submit_validation_job_request_schema.json", "r") as f:
        schema_json = json.load(f)
    return jsonify(schema_json)


@app.route("/validation_service/streaming", methods=["GET"])
def test_streaming():
    return_vals = ["A", "B", "C", "D", "E", "F", "G"]

    def generate():
        for return_val in return_vals:
            time.sleep(2)
            yield return_val

    return app.response_class(stream_with_context(generate()))


@app.route("/validation_service/submit_validation_experiment", methods=["POST"])
def validation_experiment():
    validation_request_str: str = request.form["request"]
    if validation_request_str == "":
        err_msg = "Empty request submitted"
        logger.error(err_msg)
        return build_json_response(ExperimentResponse(id="None", ok=False, err_msg=err_msg)), HTTPStatus.BAD_REQUEST

    validation_request: dict = json.loads(validation_request_str)
    pprint(validation_request)

    # Check if the POST request has the file part
    if "file" not in request.files:
        err_msg = "No file included in request"
        logger.error(err_msg)
        return build_json_response(ExperimentResponse(id="None", ok=False, err_msg=err_msg)), HTTPStatus.BAD_REQUEST

    file: FileStorage = request.files["file"]

    # If the user does not select a file, the browser submits an
    # empty file without a filename.
    if file.filename == "":
        err_msg = "Empty filename submitted"
        logger.error(err_msg)
        return build_json_response(ExperimentResponse(id="None", ok=False, err_msg=err_msg)), HTTPStatus.BAD_REQUEST

    if file is not None:
        logger.info(f"file.filename: {file.filename}")
        if allowed_file(file.filename):
            file_bytes: bytes = file.read()

            hasher = hashlib.md5()
            hasher.update(file_bytes)
            md5_hash: str = hasher.hexdigest()
            logger.info(f"Uploaded file of size {len(file_bytes)} bytes, and hash: {md5_hash}")

            with grpc.insecure_channel(f"{app.config['MASTER_HOSTNAME']}:{app.config['MASTER_PORT']}") as channel:
                stub: validation_pb2_grpc.MasterStub = validation_pb2_grpc.MasterStub(channel)

                # Build and log gRPC request
                validation_grpc_request: ValidationJobRequest = Parse(validation_request_str, ValidationJobRequest())
                validation_grpc_request.model_file.type = file_type(file.filename)
                validation_grpc_request.model_file.md5_hash = md5_hash
                validation_grpc_request.model_file.data = file_bytes

                # Log request
                logger.info(validation_grpc_request)

                # Submit validation experiment job
                experiment_grpc_response: ExperimentResponse = stub.SubmitExperiment(validation_grpc_request)
                logger.info(f"Experiment Response received: {experiment_grpc_response}")

            response_code: int = HTTPStatus.OK if experiment_grpc_response.ok else HTTPStatus.INTERNAL_SERVER_ERROR
            return build_json_response(experiment_grpc_response), response_code

        else:
            err_msg = f"File extension not allowed! Please upload only .zip, .pth, .pkl, or .h5 files"
            logger.error(err_msg)
            return build_json_response(ExperimentResponse(id="None", ok=False, err_msg=err_msg)), \
                HTTPStatus.BAD_REQUEST

    else:
        err_msg = f"Uploaded file object is None! Please upload a valid file"
        logger.error(err_msg)
        return build_json_response(ExperimentResponse(id="None", ok=False, err_msg=err_msg)), \
            HTTPStatus.BAD_REQUEST


@app.route("/validation_service/submit_validation_job", methods=["POST"])
def validation():
    def generate():
        logger.info(request.files)
        logger.info(request.data)
        logger.info(request.form)

        validation_request_str: str = request.form["request"]
        if validation_request_str == "":
            err_msg = "Empty request submitted"
            logger.error(err_msg)
            return json.dumps({
                "id": "None",
                "ok": False,
                "err_msg": err_msg
            }), HTTPStatus.BAD_REQUEST

        validation_request: dict = json.loads(validation_request_str)
        logger.info(validation_request)
        ok, err_msg = validate_request_json(validation_request)
        if not ok:
            return json.dumps({
                "id": "None",
                "ok": False,
                "err_msg": err_msg
            }), HTTPStatus.BAD_REQUEST

        # Check if the POST request has the file part
        if "file" not in request.files:
            err_msg = "No file included in request"
            logger.error(err_msg)
            return json.dumps({
                "id": "None",
                "ok": False,
                "err_msg": err_msg
            }), HTTPStatus.BAD_REQUEST

        file: FileStorage = request.files["file"]

        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == "":
            err_msg = "Empty filename submitted"
            logger.error(err_msg)
            return json.dumps({
                "id": "None",
                "ok": False,
                "err_msg": err_msg
            }), HTTPStatus.BAD_REQUEST

        if file is not None:
            if allowed_file(file.filename):
                file_bytes: bytes = file.read()

                hasher = hashlib.md5()
                hasher.update(file_bytes)
                md5_hash: str = hasher.hexdigest()
                logger.info(f"Uploaded file of size {len(file_bytes)} bytes, and hash: {md5_hash}")

                with grpc.insecure_channel(f"{app.config['MASTER_HOSTNAME']}:{app.config['MASTER_PORT']}") as channel:
                    stub: validation_pb2_grpc.MasterStub = validation_pb2_grpc.MasterStub(channel)

                    # Build and log gRPC request
                    validation_grpc_request: ValidationJobRequest = Parse(validation_request_str,
                                                                          ValidationJobRequest())
                    validation_grpc_request.model_file.type = file_type(file.filename)
                    validation_grpc_request.model_file.md5_hash = md5_hash
                    validation_grpc_request.model_file.data = file_bytes

                    # Log request
                    # logger.info(validation_grpc_request)
                    logger.info("REQUEST INFO ===============================================")
                    logger.info(f"model_file.type: {ModelFileType.Name(validation_grpc_request.model_file.type)}")
                    logger.info(f"model_file.md5_hash: {md5_hash}")
                    logger.info(f"model_file.data.length: {len(file_bytes)} bytes")
                    logger.info(f"mongo_host: {validation_grpc_request.mongo_host}")
                    logger.info(f"mongo_port: {validation_grpc_request.mongo_port}")
                    logger.info(f"read_config: {validation_grpc_request.read_config}")
                    logger.info(f"database: {validation_grpc_request.database}")
                    logger.info(f"collection: {validation_grpc_request.collection}")
                    logger.info(f"normalize_inputs: {validation_grpc_request.normalize_inputs}")
                    logger.info(f"label_field: {validation_grpc_request.label_field}")
                    logger.info(f"feature_fields: {validation_grpc_request.feature_fields}")
                    logger.info(f"model_framework: {ModelFramework.Name(validation_grpc_request.model_framework)}")
                    logger.info(f"model_category: {ModelCategory.Name(validation_grpc_request.model_category)}")
                    logger.info(f"loss_function: {LossFunction.Name(validation_grpc_request.loss_function)}")
                    logger.info(f"spatial_coverage: {SpatialCoverage.Name(validation_grpc_request.spatial_coverage)}")
                    logger.info(f"allocations: {validation_grpc_request.allocations}")
                    logger.info("==========================================================")

                    for validation_grpc_response in stub.SubmitValidationJob(validation_grpc_request):
                        logger.info(f"inside flask server!! {validation_grpc_response}")
                        dict_response = MessageToDict(validation_grpc_response, preserving_proto_field_name=True)
                        yield json.dumps(dict_response, indent=None) + '\n'
                        # yield build_json_response(validation_grpc_response)

                    # Submit validation job
                    # Needs to stream back to the client
                    # validation_grpc_response: ValidationJobResponse = stub.SubmitValidationJob(validation_grpc_request)
                    # logger.info(f"Validation Response received: {validation_grpc_response}")

                    # def generate_response():
                    #     for validation_grpc_response in stub.SubmitValidationJob(validation_grpc_request):
                    #         logger.info(f"inside flask server!! {validation_grpc_response}")
                    #         response_code: int = HTTPStatus.OK if validation_grpc_response.ok else HTTPStatus.INTERNAL_SERVER_ERROR
                    #         yield build_json_response(validation_grpc_response), response_code
                    #
                    # return app.response_class(stream_with_context(generate_response()))
                    return json.dumps({"ok": True}, indent=None), HTTPStatus.OK

            else:
                err_msg = f"File extension not allowed! Please upload only .zip, .pth, .pickle, or .h5 files"
                logger.error(err_msg)
                return json.dumps({
                    "id": "None",
                    "ok": False,
                    "err_msg": err_msg
                }), HTTPStatus.BAD_REQUEST

        else:
            err_msg = f"Uploaded file object is None! Please upload a valid file"
            logger.error(err_msg)
            return json.dumps({
                "id": "None",
                "ok": False,
                "err_msg": err_msg
            }), HTTPStatus.BAD_REQUEST

    return app.response_class(stream_with_context(generate()))


def build_json_response(grpc_msg) -> str:
    return MessageToJson(grpc_msg, preserving_proto_field_name=True)


def validate_request_json(request_json: dict) -> (bool, str):
    """Validates a JSON request in the form of a Python object.
        Returns a bool ok and a str msg"""
    schema: dict = get_schema()
    try:
        validate(instance=request_json, schema=schema)
    except jsonschema.exceptions.ValidationError as err:
        err_msg = f"Given JSON data is Invalid: {err.message}"
        logger.error(err_msg)
        return False, err_msg

    logger.info("JSON request is valid")
    return True, ""


def get_schema() -> dict:
    """Loads the given schema from resources/"""
    with open("resources/submit_validation_job_request_schema.json", "r") as file:
        schema: dict = json.load(file)
    return schema
