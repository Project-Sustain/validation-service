import requests
import json
from pprint import pprint


url = "lattice-150.cs.colostate.edu:5000/validation_service/submit_validation_job"
request_file = "/s/parsons/b/others/sustain/SustainProject/validation-service/testing/test_requests/test_request_all_gis_joins.json"
model_file = "/s/parsons/b/others/sustain/SustainProject/validation-service/testing/test_models/tensorflow/linear_regression/hdf5/my_model.h5"

with open(request_file, "r") as rfile:
    request = json.load(rfile)

payload = {"request": json.dumps(request)}

pprint(payload)

# files = [
#   ('file', ('my_model.h5',
#             open(model_file, 'rb'),
#             'application/octet-stream')
#    )
# ]
#
# headers = {}
#
# response = requests.request("POST", url, headers=headers, data=payload, files=files)
#
# print(response.text)