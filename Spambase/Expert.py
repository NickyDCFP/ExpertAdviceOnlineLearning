import vowpalwabbit as vw
import pandas as pd
from constants import EXPERT_METRICS
import re

class Expert:
    def __init__(self, features: list[str], learning_rate: float, loss_function: str):
        self.model: vw.Workspace = vw.Workspace(
            P=1,
            enable_logging=True,
            quiet=False,
            l=learning_rate,
            loss_function=loss_function
        )
        self.features: list[str] = features
        self.weight_df: pd.DataFrame = pd.DataFrame(columns=features)

    def learn(self, ex: str) -> None:
        self.model.learn(ex)
        weights: list[str] = [self.model.get_weight_from_name(feature) for feature in self.features]
        self.weight_df.loc[len(self.weight_df)] = weights

    def get_weight_history(self) -> pd.DataFrame:
        return self.weight_df

    def get_log(self) -> tuple[list[str], pd.DataFrame]:
        self.model.finish()
        logs: list[str] = self.model.get_log()
        history: pd.DataFrame = pd.DataFrame(columns=EXPERT_METRICS)
        for log in logs:
            if re.match("^[\d|.| |-]+$", log) and log.replace(" ", "") != "":
                history_parsed: list[str] = re.findall("[\d|.|-][\d|.|-]*", log)
                history.loc[len(history)] = [float(metric) for metric in history_parsed]
        return logs, history