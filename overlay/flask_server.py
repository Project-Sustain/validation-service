import json
import grpc
from flask import Flask, request
from http import HTTPStatus
from pprint import pprint
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
    app.run()  # Entrypoint


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
    # if file and allowed_file(file.filename):
    #     filename = secure_filename(file.filename)
    #     file.save(os.path.join(app.config['UPLOAD_DIR'], filename))
    #     return f'File {filename} successfully saved', HTTPStatus.OK

    # Create gRPC request to master node
    master_hostname = "inf0rmatiker-desktop"
    master_port = 50051
    with grpc.insecure_channel(f'{master_hostname}:{master_port}') as channel:
        stub = validation_pb2_grpc.MasterStub(channel)
        file_upload_response = stub.UploadFile(file_chunker.chunk_file(file))
    info(f"Response received: {file_upload_response}")
