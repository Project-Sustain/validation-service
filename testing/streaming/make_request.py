import http.client
import mimetypes
from codecs import encode
import json
import sys
import os
import requests
from pprint import pprint

model_file = f"model.h5"
request_file = f"request.json"

if not os.path.exists(request_file):
    print(f"{request_file} does not exist! Please create first (hint, use the testing/test_requests/ JSON requests as a template example)")
    exit(1)

url = "http://lattice-150.cs.colostate.edu:5000/validation_service/submit_validation_job"


with open(request_file, "r") as rfile:
    request = json.load(rfile)

files = [
  ('file', (model_file, open('model.h5', 'rb')))
]
payload = {
    "request": json.dumps(request)
}

pprint(requests.Request('POST', url, data=payload, files=files).prepare().body)

response = requests.request("POST", url, data=payload, files=files, stream=True)
for line in response.iter_lines():
    pprint(line)

# print("RECEIVED RESPONSE")
# print(response.text.encode('utf8'))
#
# conn = http.client.HTTPConnection("lattice-150.cs.colostate.edu", 5000)
# dataList = []
# boundary = 'wL36Yn8afVp8Ag7AmP8qZ0SA4n1v9T'
# dataList.append(encode('--' + boundary))
# dataList.append(encode(f"Content-Disposition: form-data; name=file; filename={model_file}"))
#
# fileType = mimetypes.guess_type(model_file)[0] or 'application/octet-stream'
# dataList.append(encode('Content-Type: {}'.format(fileType)))
# dataList.append(encode(''))
#
# with open(model_file, 'rb') as f:
#   dataList.append(f.read())
# dataList.append(encode('--' + boundary))
# dataList.append(encode('Content-Disposition: form-data; name=request;'))
#
# dataList.append(encode('Content-Type: {}'.format('text/plain')))
# dataList.append(encode(''))
# dataList.append(encode(json.dumps(request)))
#
# dataList.append(encode('--'+boundary+'--'))
# dataList.append(encode(''))
# body = b'\r\n'.join(dataList)
# payload = body
# headers = {
#    'Content-type': 'multipart/form-data; boundary={}'.format(boundary)
# }
# conn.request("POST", "/validation_service/submit_validation_job", payload, headers)
# res = conn.getresponse()
# data = res.read()
# print(data.decode("utf-8"))
