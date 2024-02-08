from Expert import Expert
import numpy as np
import pandas as pd
from constants import METRICS, LABEL_FORMAT_INDEX
class ExpertAdvice:
    def __init__(self, experts: list[Expert], alpha: float = 0, learning_rate: float = 0.5):
        """
        Constructor for the ExpertsAdvice class.

        Parameters:
            experts:        the ExpertAdvice's pre-initialized online learner. Do not need to be
                            pre-trained.
            alpha:          the sharing ratio for the fixed-share algorithm. 
                            Set to 0 for static-expert.
            learning_rate:  the learning rate for the learner.
        """
        self.experts: list[Expert] = experts
        n = len(experts)
        self.alpha: float = alpha
        self.transition_matrix: np.ndarray = np.eye(n) * (1 - alpha)
        self.transition_matrix -= (np.eye(n) - np.ones(n)) * (alpha / (n - 1))
        self.learning_rate: float = learning_rate
        self.p: np.ndarray = np.array([1/len(self.experts) for _ in experts])
        self.history: pd.DataFrame = pd.DataFrame(columns=METRICS)
        self.cumulative_loss: float = 0
        self.sample_number: int = 0
        self.weight_df: pd.DataFrame = pd.DataFrame(
            [self.p],
            columns=[f"Expert {i + 1}" for i in range(n)]
        )

    def predict(self, ex: str) -> float:
        """
        Predicts on an example and returns the prediction.

        Parameters:
            ex:     the example to be predicted on.
        
        Returns:
            the learner's prediction on the example.
        """
        predictions: np.ndarray = np.array([expert.predict(ex) for expert in self.experts])
        return predictions.dot(self.p)
    
    def learn(self, ex: str) -> None:
        """
        Predicts on an example, prompts the experts to learn, and updates the learner weights.

        Parameters:
            ex:     the example to be learned from.
        """
        expert_outputs: pd.DataFrame = pd.DataFrame(columns=METRICS)
        for expert in self.experts:
            expert_outputs.loc[len(expert_outputs)] = expert.learn(ex)
        expert_predictions: np.ndarray = expert_outputs["predicted label"].to_numpy()
        y_pred: float = expert_predictions.dot(self.p)
        y: float = (int(ex[:LABEL_FORMAT_INDEX]) + 1) / 2
        loss: float = -(
            y * (np.log(y_pred) if y_pred != 0 else 0) +
            (1 - y) * (np.log(1 - y_pred) if y_pred != 1 else 0))
        if np.isnan(loss):
            loss = 0
        self.cumulative_loss += loss
        self.sample_number += 1
        self.history.loc[len(self.history)] = [
            y,
            y_pred,
            self.cumulative_loss,
            self.cumulative_loss / self.sample_number,
            loss
        ]
        expert_losses: np.ndarray = expert_outputs["sample loss"].to_numpy()
        emission_probabilities: np.ndarray = np.exp(-self.learning_rate * expert_losses)
        self.p = self.transition_matrix.dot(emission_probabilities * self.p)
        z: float = np.sum(self.p)
        self.p /= z
        self.weight_df.loc[len(self.weight_df)] = self.p.T
    
    def get_log(self) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
        """
        Retrieves the experts' logs as well as the learner's history DataFrame.

        Returns:
            The learner's accuracy metrics at every timestep and the experts' accuracy metrics
            at every timestep.
        """
        return self.history, [expert.get_log() for expert in self.experts]

    def get_weight_history(self) -> pd.DataFrame:
        """
        Retrieves the evolution of the learner and expert weights over time.

        Returns:
            The learner's weights at every timestep and the experts' weights at every timestep.
        """
        return self.weight_df, [expert.get_weight_history() for expert in self.experts]