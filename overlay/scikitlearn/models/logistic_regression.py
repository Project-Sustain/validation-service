import pickle

from overlay.validation_pb2 import ValidationJobRequest


class LogisticRegression:
    def __init__(self):
        self.model = None

    def load_model(self, model_path):
        self.model = pickle.load(open(model_path, 'rb'))

    def validate(self, validation_request: ValidationJobRequest):
        pass
