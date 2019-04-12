
class Classifier:
    def read_data(self):
        raise NotImplementedError()

    def get_classifier(self):
        raise NotImplementedError()

    def predict(self):
        raise NotImplementedError()