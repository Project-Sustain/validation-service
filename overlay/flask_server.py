import json
import grpc
import hashlib
import jsonschema
from jsonschema import validate
from flask import Flask, request, jsonify
from http import HTTPStatus
from pprint import pprint
from logging import info, error
from google.protobuf.json_format import MessageToJson, Parse
from werkzeug.datastructures import FileStorage

from overlay import validation_pb2_grpc
from overlay.validation_pb2 import ValidationJobRequest, ValidationJobResponse, ModelFileType, ExperimentResponse

UPLOAD_DIR = "./uploads"
ALLOWED_EXTENSIONS = {"zip", "pt", "pkl", "h5"}

app = Flask(__name__)
app.config["UPLOAD_DIR"] = UPLOAD_DIR


# Main entrypoint
def run(master_hostname="localhost", master_port=50051, flask_port=5000):
    print("Running flask server...")
    app.config["MASTER_HOSTNAME"] = master_hostname
    app.config["MASTER_PORT"] = master_port
    app.run(host="0.0.0.0", port=flask_port)  # Entrypoint


def allowed_file(filename) -> bool:
    return "." in filename and \
           filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def file_type(filename) -> ModelFileType:
    extension = filename.rsplit(".", 1)[1].lower()
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
    info(f"Received GET request for ")
    with open("./resources/submit_validation_job_request_schema.json", "r") as f:
        schema_json = json.load(f)
    return jsonify(schema_json)


@app.route("/validation_service/submit_validation_experiment", methods=["POST"])
def validation_experiment():
    validation_request_str: str = request.form["request"]
    if validation_request_str == "":
        err_msg = "Empty request submitted"
        error(err_msg)
        return build_json_response(ExperimentResponse(id="None", ok=False, err_msg=err_msg)), HTTPStatus.BAD_REQUEST

    validation_request: dict = json.loads(validation_request_str)
    pprint(validation_request)

    # Check if the POST request has the file part
    if "file" not in request.files:
        err_msg = "No file included in request"
        error(err_msg)
        return build_json_response(ExperimentResponse(id="None", ok=False, err_msg=err_msg)), HTTPStatus.BAD_REQUEST

    file: FileStorage = request.files["file"]

    # If the user does not select a file, the browser submits an
    # empty file without a filename.
    if file.filename == "":
        err_msg = "Empty filename submitted"
        error(err_msg)
        return build_json_response(ExperimentResponse(id="None", ok=False, err_msg=err_msg)), HTTPStatus.BAD_REQUEST

    if file is not None:
        if allowed_file(file.filename):
            file_bytes: bytes = file.read()

            hasher = hashlib.md5()
            hasher.update(file_bytes)
            md5_hash: str = hasher.hexdigest()
            info(f"Uploaded file of size {len(file_bytes)} bytes, and hash: {md5_hash}")

            with grpc.insecure_channel(f"{app.config['MASTER_HOSTNAME']}:{app.config['MASTER_PORT']}") as channel:
                stub: validation_pb2_grpc.MasterStub = validation_pb2_grpc.MasterStub(channel)

                # Build and log gRPC request
                validation_grpc_request: ValidationJobRequest = Parse(validation_request_str, ValidationJobRequest())
                validation_grpc_request.model_file.type = file_type(file.filename)
                validation_grpc_request.model_file.md5_hash = md5_hash
                validation_grpc_request.model_file.data = file_bytes

                info(validation_grpc_request)

                # Submit validation experiment job
                experiment_grpc_response: ExperimentResponse = stub.SubmitExperiment(validation_grpc_request)
                info(f"Experiment Response received: {experiment_grpc_response}")

            response_code: int = HTTPStatus.OK if experiment_grpc_response.ok else HTTPStatus.INTERNAL_SERVER_ERROR
            return build_json_response(experiment_grpc_response), response_code

        else:
            err_msg = f"File extension not allowed! Please upload only .zip, .pth, .pickle, or .h5 files"
            error(err_msg)
            return build_json_response(ExperimentResponse(id="None", ok=False, err_msg=err_msg)), \
                HTTPStatus.BAD_REQUEST

    else:
        err_msg = f"Uploaded file object is None! Please upload a valid file"
        error(err_msg)
        return build_json_response(ExperimentResponse(id="None", ok=False, err_msg=err_msg)), \
            HTTPStatus.BAD_REQUEST


@app.route("/validation_service/submit_validation_job", methods=["POST"])
def validation():
    validation_request_str: str = request.form["request"]
    if validation_request_str == "":
        err_msg = "Empty request submitted"
        error(err_msg)
        return build_json_response(ValidationJobResponse(id="None", ok=False, err_msg=err_msg)), HTTPStatus.BAD_REQUEST

    validation_request: dict = json.loads(validation_request_str)
    info(validation_request)
    ok, err_msg = validate_request_json(validation_request)
    if not ok:
        return build_json_response(ValidationJobResponse(id="None", ok=False, err_msg=err_msg)), HTTPStatus.BAD_REQUEST

    # Check if the POST request has the file part
    if "file" not in request.files:
        err_msg = "No file included in request"
        error(err_msg)
        return build_json_response(ValidationJobResponse(id="None", ok=False, err_msg=err_msg)), HTTPStatus.BAD_REQUEST

    file: FileStorage = request.files["file"]

    # If the user does not select a file, the browser submits an
    # empty file without a filename.
    if file.filename == "":
        err_msg = "Empty filename submitted"
        error(err_msg)
        return build_json_response(ValidationJobResponse(id="None", ok=False, err_msg=err_msg)), HTTPStatus.BAD_REQUEST

    if file is not None:
        if allowed_file(file.filename):
            file_bytes: bytes = file.read()

            hasher = hashlib.md5()
            hasher.update(file_bytes)
            md5_hash: str = hasher.hexdigest()
            info(f"Uploaded file of size {len(file_bytes)} bytes, and hash: {md5_hash}")

            with grpc.insecure_channel(f"{app.config['MASTER_HOSTNAME']}:{app.config['MASTER_PORT']}") as channel:
                stub: validation_pb2_grpc.MasterStub = validation_pb2_grpc.MasterStub(channel)

                # Build and log gRPC request
                validation_grpc_request: ValidationJobRequest = Parse(validation_request_str, ValidationJobRequest())
                validation_grpc_request.model_file.type = file_type(file.filename)
                validation_grpc_request.model_file.md5_hash = md5_hash
                validation_grpc_request.model_file.data = file_bytes

                info(validation_grpc_request)

                # Submit validation job
                validation_grpc_response: ValidationJobResponse = stub.SubmitValidationJob(validation_grpc_request)
                info(f"Validation Response received: {validation_grpc_response}")

            response_code: int = HTTPStatus.OK if validation_grpc_response.ok else HTTPStatus.INTERNAL_SERVER_ERROR
            return build_json_response(validation_grpc_response), response_code

        else:
            err_msg = f"File extension not allowed! Please upload only .zip, .pth, .pickle, or .h5 files"
            error(err_msg)
            return build_json_response(ValidationJobResponse(id="None", ok=False, err_msg=err_msg)), \
                HTTPStatus.BAD_REQUEST

    else:
        err_msg = f"Uploaded file object is None! Please upload a valid file"
        error(err_msg)
        return build_json_response(ValidationJobResponse(id="None", ok=False, err_msg=err_msg)), \
            HTTPStatus.BAD_REQUEST


def build_json_response(validation_grpc_response: ValidationJobResponse) -> str:
    return MessageToJson(validation_grpc_response, preserving_proto_field_name=True)


def validate_request_json(request_json: dict) -> (bool, str):
    """Validates a JSON request in the form of a Python object.
        Returns a bool ok and a str msg"""
    schema: dict = get_schema()
    try:
        validate(instance=request_json, schema=schema)
    except jsonschema.exceptions.ValidationError as err:
        err_msg = f"Given JSON data is Invalid: {err.message}"
        error(err_msg)
        return False, err_msg

    info("JSON request is valid")
    return True, ""


def get_schema() -> dict:
    """Loads the given schema from resources/"""
    with open("resources/submit_validation_job_request_schema.json", "r") as file:
        schema: dict = json.load(file)
    return schema
