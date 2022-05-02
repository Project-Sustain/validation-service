import http.client
import mimetypes
from codecs import encode
import json
import sys
import os

experiment_dir = sys.argv[1]
experiment_request = f"{experiment_dir}/request.json"
experiment_dir_parts = experiment_dir.split("/")
exp_framework_path = experiment_dir_parts[0]  # i.e. pytorch, tensorflow, scikitlearn
exp_model_category_path = f"{exp_framework_path}/{experiment_dir_parts[1]}"
dir_walk = os.walk(exp_model_category_path)
model_file = next(dir_walk)[2][0]
model_file = f"{exp_model_category_path}/{model_file}"

if not os.path.exists(experiment_request):
    print(f"{experiment_request} does not exist! Please create first (hint, use the testing/test_requests/ JSON requests as a template example)")
    exit(1)

url = "lattice-150.cs.colostate.edu:5000/validation_service/submit_validation_experiment"
request_file = f"/s/parsons/b/others/sustain/SustainProject/validation-service/benchmarks/{experiment_dir}/request.json"

with open(request_file, "r") as rfile:
    request = json.load(rfile)

conn = http.client.HTTPConnection("lattice-150.cs.colostate.edu", 5000)
dataList = []
boundary = 'wL36Yn8afVp8Ag7AmP8qZ0SA4n1v9T'
dataList.append(encode('--' + boundary))
dataList.append(encode(f"Content-Disposition: form-data; name=file; filename={model_file}"))

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
conn.request("POST", "/validation_service/submit_validation_experiment", payload, headers)
res = conn.getresponse()
data = res.read()
print(data.decode("utf-8"))
