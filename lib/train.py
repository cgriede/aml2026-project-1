from sklearn.linear_model import Ridge

class Trainer:
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def train_baseline(self):
        """Use a simple Ridge model to get a baseline score"""
        model = Ridge(alpha=1.0)
        model.fit(self.X, self.y)
        return model
