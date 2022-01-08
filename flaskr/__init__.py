import os
from flask import Flask, request
from werkzeug.utils import secure_filename
from http import HTTPStatus

UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'zip'}

app = Flask(__name__)
app.config['UPLOAD_DIR'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def hello():
    return 'Hello, World!'


@app.route('/validation', methods=['POST'])
def validation():

    # Check if the POST request has the file part
    if 'file' not in request.files:
        return 'No file included in request', HTTPStatus.BAD_REQUEST

    file = request.files['file']

    # If the user does not select a file, the browser submits an
    # empty file without a filename.
    if file.filename == '':
        return 'Empty file submitted', HTTPStatus.BAD_REQUEST

    # Save file
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_DIR'], filename))
        return f'File {filename} successfully saved', HTTPStatus.OK
