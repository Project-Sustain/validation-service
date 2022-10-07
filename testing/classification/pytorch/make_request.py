import http.client
import mimetypes
from codecs import encode
import json
import sys
import os
import requests
from pprint import pprint
import time


class Timer:

    def __init__(self, func=time.perf_counter):
        self.elapsed = 0.0
        self._func = func
        self._start = None

    # starting the module
    def start(self):
        if self._start is not None:
            raise RuntimeError('Already started')
        self._start = self._func()

    # stopping the timer
    def stop(self):
        if self._start is None:
            raise RuntimeError('Not started')
        end = self._func()
        self.elapsed += end - self._start
        self._start = None

    # resetting the timer
    def reset(self):
        self.elapsed = 0.0

    def running(self):
        return self._start is not None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


model_file = f"model.pt"
request_file = f"request.json"

if not os.path.exists(request_file):
    print(
        f"{request_file} does not exist! Please create first (hint, use the testing/test_requests/ JSON requests as a template example)")
    exit(1)

url = "http://lattice-100.cs.colostate.edu:5000/validation_service/submit_validation_job"

with open(request_file, "r") as rfile:
    request = json.load(rfile)

files = [
    ('file', (model_file, open(
        '/s/parsons/b/others/sustain/SustainProject/validation-service/testing/classification/pytorch/model.pt',
        'rb')))
]
payload = {
    "request": json.dumps(request)
}

count: int = 0
profiler: Timer = Timer()
profiler.start()

response = requests.request("POST", url, data=payload, files=files, stream=True)
response_timestamps = []
for line in response.iter_lines():
    pprint(line)
    profiler.stop()
    response_timestamps.append(profiler.elapsed)
    profiler.start()
    data = json.loads(line)
    pprint(data)

profiler.stop()
results = {
    "start_sec": 0,
    "stop_sec": profiler.elapsed,
    "result_arrivals": response_timestamps
}

with open("streaming_results.json", "w") as f:
    json.dump(results, f)
