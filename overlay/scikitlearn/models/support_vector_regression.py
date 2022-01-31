import pickle

from overlay.scikitlearn.wireformats.validation_request import ValidationRequest


class SupportVectorRegression:
    def __init__(self):
        self.model = None

    def load_model(self, model_path):
        self.model = pickle.load(open(model_path, 'rb'))

    def validate(self, validation_requet: ValidationRequest):
        X_test = validation_requet.independent_vars
        y_test = validation_requet.dependent_vars
        score = self.model.score(X_test, y_test)
        return validation_requet.gis_join, score
