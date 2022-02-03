import json
import grpc
import os
import hashlib
from flask import Flask, request
from http import HTTPStatus
from pprint import pprint

from werkzeug.utils import secure_filename

import filereader
import validation_pb2
import validation_pb2_grpc

from logging import info

UPLOAD_DIR = './uploads'
ALLOWED_EXTENSIONS = {'zip'}

app = Flask(__name__)
app.config['UPLOAD_DIR'] = UPLOAD_DIR


# Main entrypoint
def run(master_hostname="localhost", master_port=50051, flask_port=5000):
    print("Running flask server...")
    app.config['MASTER_HOSTNAME'] = master_hostname
    app.config['MASTER_PORT'] = master_port
    app.run(host="0.0.0.0", port=flask_port)  # Entrypoint


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def hello():
    return 'Hello, World!'


@app.route('/validation', methods=['POST'])
def validation():
    validation_request_str = request.form["request"]
    if validation_request_str == '':
        return 'Empty request submitted', HTTPStatus.BAD_REQUEST

    validation_request = json.loads(validation_request_str)
    pprint(validation_request)

    # Check if the POST request has the file part
    if 'file' not in request.files:
        return 'No file included in request', HTTPStatus.BAD_REQUEST

    file = request.files['file']

    # If the user does not select a file, the browser submits an
    # empty file without a filename.
    if file.filename == '':
        return 'Empty file submitted', HTTPStatus.BAD_REQUEST

    if file and allowed_file(file.filename):
        file_bytes = file.read()

        hasher = hashlib.md5()
        hasher.update(file_bytes)
        md5_hash = hasher.hexdigest()
        info(f"Uploaded file of size {len(file_bytes)} bytes, and hash: {md5_hash}")

        with grpc.insecure_channel(f"{app.config['MASTER_HOSTNAME']}:{app.config['MASTER_PORT']}") as channel:
            stub = validation_pb2_grpc.MasterStub(channel)
            model_file = validation_pb2.ModelFile(
                type="zip",
                md5_hash=md5_hash,
                data=file_bytes
            )
            validation_grpc_request = validation_pb2.ValidationJobRequest(
                model_framework=validation_request.model_framework,
                model_type=validation_request.model_type,
                database=validation_request.database,
                collection=validation_request.collection,
                spatial_field=validation_request.spatial_field,
                label_field=validation_request.label_field,
                validation_metric=validation_request.validation_metric,
                feature_fields=validation_request.feature_fields,
                gis_joins=validation_request.gis_joins,
                model_file=model_file
            )

            validation_grpc_response = stub.SubmitValidationJob(validation_grpc_request)
            info(f"Validation Response received: {validation_grpc_response}")

    return f"ValidationJobResponse: {validation_grpc_response}", HTTPStatus.OK
