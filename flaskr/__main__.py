import os
import json
from flask import Flask, request
from werkzeug.utils import secure_filename
from http import HTTPStatus
from pprint import pprint

UPLOAD_DIR = './uploads'
RESOURCES_DIR = './flaskr/resources'
ALLOWED_EXTENSIONS = {'zip'}
COUNTY_GISJOINS = []

app = Flask(__name__)
app.config['UPLOAD_DIR'] = UPLOAD_DIR


# Main entrypoint
def main():
    print("Running main() function...")
    global COUNTY_GISJOINS
    COUNTY_GISJOINS = load_gis_joins()
    print(f"Loaded {len(COUNTY_GISJOINS)} county GISJOINs")
    app.run()  # Entrypoint


def load_gis_joins():
    county_gis_joins = []
    gis_join_filename = f"{RESOURCES_DIR}/gis_joins.json"
    with open(gis_join_filename, "r") as read_file:
        print("Loading in list of county GISJOINs...")
        gis_joins = json.load(read_file)
        states = gis_joins["states"]
        for state_key, state_value in states.items():
            for county_key, county_value in state_value["counties"].items():
                county_gis_joins.append(county_value["GISJOIN"])

    return county_gis_joins


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
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_DIR'], filename))
        return f'File {filename} successfully saved', HTTPStatus.OK


if __name__ == "__main__":
    main()