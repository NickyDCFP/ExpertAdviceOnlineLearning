import vowpalwabbit as vw
import pandas as pd
from constants import METRICS, LABEL_FORMAT_INDEX, LOGISTIC
import numpy as np
import re

class Expert:
    def __init__(self, features: list[str], learning_rate: float):
        self.model: vw.Workspace = vw.Workspace(
            P=1,
            enable_logging=True,
            quiet=False,
            l=learning_rate,
            loss_function=LOGISTIC,
            link=LOGISTIC
        )
        self.features: list[str] = features
        self.weight_df: pd.DataFrame = pd.DataFrame(columns=features)
        self.history: pd.DataFrame = pd.DataFrame(columns=METRICS)
        self.cumulative_loss: float = 0
        self.sample_number: int = 0

    def learn(self, ex: str) -> None:
        self.log_example(ex)
        self.model.learn(ex)
        weights: list[str] = [self.model.get_weight_from_name(feature) for feature in self.features]
        self.weight_df.loc[len(self.weight_df)] = weights
        return self.history.iloc[-1]
    
    def log_example(self, ex: str) -> None:
        y_pred: float = self.model.predict(ex)
        y: int = int(ex[:LABEL_FORMAT_INDEX])
        ce: float = self.cross_entropy(ex)
        self.sample_number += 1
        self.cumulative_loss += ce
        self.history.loc[len(self.history)] = [
            y,
            y_pred,
            self.cumulative_loss,
            self.cumulative_loss / self.sample_number,
            ce
        ]

    def predict(self, ex: str) -> int:
        return self.model.predict(ex)
    
    def cross_entropy(self, ex: str) -> float:
        y_pred: float = self.model.predict(ex)
        y: float = (int(ex[:LABEL_FORMAT_INDEX]) + 1) / 2
        loss: float = -(
            y * (np.log(y_pred) if y_pred != 0 else 0) +
            (1 - y) * (np.log(1 - y_pred) if y_pred != 1 else 0))
        if np.isnan(loss):
            return 0
        return loss

    def get_weight_history(self) -> pd.DataFrame:
        return self.weight_df

    def get_log(self) -> tuple[list[str], pd.DataFrame]:
        self.model.finish()
        return self.model.get_log(), self.history