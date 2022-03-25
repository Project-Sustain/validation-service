import json
import grpc
import hashlib
from flask import Flask, request
from http import HTTPStatus
from pprint import pprint
from logging import info, error
from google.protobuf.json_format import MessageToJson, Parse
from werkzeug.datastructures import FileStorage

from overlay import validation_pb2_grpc
from overlay.validation_pb2 import ValidationJobRequest, ValidationJobResponse, ModelFileType

UPLOAD_DIR = "./uploads"
ALLOWED_EXTENSIONS = {"zip", "pth", "pkl", "h5"}

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
    if extension == "pth":
        return ModelFileType.PYTORCH_ZIPFILE
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
    return "Welcome to the Sustain Validation Service"


@app.route("/validation_service/submit_validation_job", methods=["POST"])
def validation():
    validation_request_str: str = request.form["request"]
    if validation_request_str == "":
        err_msg = "Empty request submitted"
        error(err_msg)
        return build_json_response(ValidationJobResponse(id="None", ok=False, err_msg=err_msg)), HTTPStatus.BAD_REQUEST

    validation_request: dict = json.loads(validation_request_str)
    pprint(validation_request)

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
