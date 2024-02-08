"""
A basic online logistic regression model built over the Vowpal Wabbit (VW) online
learning framework.
"""
import vowpalwabbit as vw
import pandas as pd
from constants import METRICS, LABEL_FORMAT_INDEX, LOGISTIC
import numpy as np
import re

class Expert:
    def __init__(self, features: list[str], learning_rate: float):
        """
        Constructor for the Expert class.

        Parameters:
            features:       the list of features that the expert will predict on
            learning_rate:  the learning rate of the expert
        """
        self.model: vw.Workspace = vw.Workspace(
            P=1,
            enable_logging=True,
            quiet=False,
            l=learning_rate,
            loss_function=LOGISTIC,
            link=LOGISTIC
        )
        self.learning_rate = learning_rate
        self.features: list[str] = features
        self.weight_df: pd.DataFrame = pd.DataFrame(columns=features)
        self.history: pd.DataFrame = pd.DataFrame(columns=METRICS)
        self.cumulative_loss: float = 0
        self.sample_number: int = 0

    def learn(self, ex: str) -> pd.Series:
        """
        Learns one training example and returns the model's accuracy metrics on the sample.

        Parameters:
            ex:     the example to be learned

        Returns:
            The model's prediction and loss metrics on the sample.
        """
        self.log_example(ex)
        self.model.learn(ex)
        weights: list[str] = [self.model.get_weight_from_name(feature) for feature in self.features]
        self.weight_df.loc[len(self.weight_df)] = weights
        return self.history.iloc[-1]
    
    def log_example(self, ex: str) -> None:
        """
        Predicts on an example and logs the accuracy metrics in the history DataFrame.

        Parameters:
            ex:     the example to be predicted on
        """
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
        """
        Predicts on an example and returns the prediction.

        Parameters:
            ex:     the example to be predicted on

        Returns:
            The model's prediction on ex.
        """
        return self.model.predict(ex)
    
    def cross_entropy(self, ex: str) -> float:
        """
        Predicts on an example and returns the model's cross-entropy loss against
        the example's true label.

        Parameters:
            ex:     the example to be predicted on.

        Returns:
            The model's cross-entropy loss against the example.
        """
        y_pred: float = self.model.predict(ex)
        y: float = (int(ex[:LABEL_FORMAT_INDEX]) + 1) / 2
        loss: float = -(
            y * (np.log(y_pred) if y_pred != 0 else 0) +
            (1 - y) * (np.log(1 - y_pred) if y_pred != 1 else 0))
        if np.isnan(loss):
            return 0
        return loss

    def get_weight_history(self) -> pd.DataFrame:
        """
        Retrieves the evolution of the expert's weights over training time.

        Returns:
            All of the model's weights at every timestep.
        """
        return self.weight_df

    def get_log(self) -> tuple[list[str], pd.DataFrame]:
        """
        Finishes the model and returns its history.

        Returns:
            The model's predictions and loss metrics at every timestep and the VW console output.
        """
        self.model.finish()
        return self.model.get_log(), self.history