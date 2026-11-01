class Evaluator:
    def __init__(self, model, X, y):
        self.model = model
        self.X = X
        self.y = y

    def evaluate(self):
        score = self.model.score(self.X, self.y)
        return score
