import http.client
import mimetypes
from codecs import encode
import json
import sys
import os
import requests
from pprint import pprint
import time



url = "http://lattice-100.cs.colostate.edu:5000/validation_service/testFloods"


response = requests.request("GET", url, stream=True)
for line in response.iter_lines():
    data = json.loads(line)
    pprint(data)

