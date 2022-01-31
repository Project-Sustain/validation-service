import pickle

from overlay.constants import DB_PORT, DB_HOST
from overlay.scikitlearn.db.querier import Querier
from overlay.validation_pb2 import ValidationJobRequest


class LinearRegression:
    def __init__(self):
        self.model = None

    def load_model(self, model_path):
        self.model = pickle.load(open(model_path, 'rb'))

    def validate(self, validation_request: ValidationJobRequest):
        sustaindb = Querier(f'{DB_HOST}:{DB_PORT}', 'sustaindb')
        results = sustaindb.query("county_svi", "G0100150")
        print(f'results: {results}')


if __name__ == "__main__":
    print("Starting...")
