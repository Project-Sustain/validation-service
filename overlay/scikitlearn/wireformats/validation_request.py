class ValidationRequest:
    def __init__(self, collection: str, gis_join: str, dependent_vars: list, independent_vars, model):
        self.collection = collection
        self.gis_join = gis_join
        self.dependent_vars = dependent_vars
        self.independent_vars = independent_vars
        self.model = model
