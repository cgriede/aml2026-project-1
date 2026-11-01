import pandas as pd
from pathlib import Path
import subprocess

class DataOrchestrator:
    def __init__(self, X, model):
        self.X = X
        self.model = model
        self.y_pred = None

    def predict(self):
        self.y_pred = self.model.predict(self.X)
        return self.y_pred

    def store_predictions(self):
        assert self.y_pred is not None, "Predictions not generated"
        predictions_dir = Path("predictions")
        predictions_dir.mkdir(exist_ok=True)
        predictions_path = predictions_dir / f"{self.model.__class__.__name__}.csv"
        pd.DataFrame(self.y_pred, index=self.X.index, columns=["y"]).to_csv(predictions_path)

    def submit_predictions(self):
        predictions_dir = Path("predictions")
        predictions_path = predictions_dir / f"{self.model.__class__.__name__}.csv"
        assert predictions_path.exists(), "Predictions file not found"
        subprocess.run(f"kaggle competitions submit -c eth-aml-2026-project-1 -f {predictions_path} -m \"{self.model.__class__.__name__} predictions\"", shell=True)
