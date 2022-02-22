import pickle

from overlay.validation_pb2 import ValidationJobRequest


class SupportVectorRegression:
    def __init__(self):
        self.model = None

    def load_model(self, model_path):
        self.model = pickle.load(open(model_path, 'rb'))

    def validate(self, validation_request: ValidationJobRequest):
        pass
