from plantclef.config import get_device


class FaissClassifier:
    def __init__(self, train_embeddings):
        self.train_embeddings = train_embeddings
        self.device = get_device()

    def build_index(self):
        pass

    def make_predictions(self, embeddings):
        pass
