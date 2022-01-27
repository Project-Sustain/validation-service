import json
import grpc
import os
import hashlib
from flask import Flask, request
from http import HTTPStatus
from pprint import pprint

from werkzeug.utils import secure_filename

import file_chunker
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

    # Save file
    saved_filename = "my_model.zip"
    if file and allowed_file(file.filename):
        file.save(os.path.join(app.config['UPLOAD_DIR'], saved_filename))

    # Get file hash
    with open(f"{UPLOAD_DIR}/{saved_filename}", "rb") as f:
        hasher = hashlib.md5()
        buf = f.read()
        hasher.update(buf)
    info(f"Uploaded file hash: {hasher.hexdigest()}")

    # Create gRPC request to master node
    with open(f"{UPLOAD_DIR}/{saved_filename}", "rb") as f:
        with grpc.insecure_channel(f"{app.config['MASTER_HOSTNAME']}:{app.config['MASTER_PORT']}") as channel:
            stub = validation_pb2_grpc.MasterStub(channel)
            file_upload_response = stub.UploadFile(file_chunker.chunk_file(f))

    info(f"Response received: {file_upload_response}")
    return f'File {saved_filename} successfully saved', HTTPStatus.OK
