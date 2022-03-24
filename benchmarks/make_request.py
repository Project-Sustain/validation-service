import http.client
import mimetypes
from codecs import encode
import json
from pprint import pprint

url = "lattice-150.cs.colostate.edu:5000/validation_service/submit_validation_job"
request_file = "/s/parsons/b/others/sustain/SustainProject/validation-service/testing/test_requests/test_request_all_gis_joins.json"
model_file = "/s/parsons/b/others/sustain/SustainProject/validation-service/testing/test_models/tensorflow/linear_regression/hdf5/my_model.h5"

with open(request_file, "r") as rfile:
    request = json.load(rfile)

conn = http.client.HTTPConnection("lattice-150.cs.colostate.edu", 5000)
dataList = []
boundary = 'wL36Yn8afVp8Ag7AmP8qZ0SA4n1v9T'
dataList.append(encode('--' + boundary))
dataList.append(encode('Content-Disposition: form-data; name=file; filename=my_model.h5'))

fileType = mimetypes.guess_type(model_file)[0] or 'application/octet-stream'
dataList.append(encode('Content-Type: {}'.format(fileType)))
dataList.append(encode(''))

with open(model_file, 'rb') as f:
  dataList.append(f.read())
dataList.append(encode('--' + boundary))
dataList.append(encode('Content-Disposition: form-data; name=request;'))

dataList.append(encode('Content-Type: {}'.format('text/plain')))
dataList.append(encode(''))
dataList.append(encode(json.dumps(request)))

dataList.append(encode('--'+boundary+'--'))
dataList.append(encode(''))
body = b'\r\n'.join(dataList)
payload = body
headers = {
   'Content-type': 'multipart/form-data; boundary={}'.format(boundary)
}
conn.request("POST", "/validation_service/submit_validation_job", payload, headers)
res = conn.getresponse()
data = res.read()
print(data.decode("utf-8"))