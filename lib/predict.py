import json
import subprocess
import time
from pathlib import Path

import pandas as pd

from GLOBAL import KAGGLE_COMPETITION


class DataOrchestrator:
    def __init__(self, X, model):
        self.X = X
        self.model = model
        self.y_pred = None

    def predict(self):
        self.y_pred = self.model.predict(self.X)
        return self.y_pred

    def _predictions_path(self) -> Path:
        predictions_dir = Path("predictions")
        predictions_dir.mkdir(exist_ok=True)
        return predictions_dir / f"{self.model.__class__.__name__}.csv"

    def store_predictions(self):
        assert self.y_pred is not None, "Predictions not generated"
        pd.DataFrame(self.y_pred, index=self.X.index, columns=["y"]).to_csv(
            self._predictions_path()
        )

    def submit_predictions(self):
        predictions_path = self._predictions_path()
        assert predictions_path.exists(), "Predictions file not found"
        subprocess.run(
            [
                "kaggle",
                "competitions",
                "submit",
                "-c",
                KAGGLE_COMPETITION,
                "-f",
                str(predictions_path),
                "-m",
                f"{self.model.__class__.__name__} predictions",
            ],
            check=True,
        )
        score = _wait_public_score(predictions_path.name)
        print(f"Kaggle public score: {score}")
        return score


def _wait_public_score(filename: str, timeout_s: int = 180, interval_s: int = 5) -> str:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        raw = subprocess.check_output(
            [
                "kaggle",
                "competitions",
                "submissions",
                KAGGLE_COMPETITION,
                "--format",
                "json",
                "--page-size",
                "5",
            ],
            text=True,
        )
        rows = json.loads(raw)
        row = next((r for r in rows if r.get("fileName") == filename), rows[0] if rows else None)
        if row and row.get("publicScore"):
            return str(row["publicScore"])
        time.sleep(interval_s)
    raise TimeoutError(f"No public score for {filename} after {timeout_s}s")
