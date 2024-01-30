import pandas as pd
from fileio import read_df_from_file, convert_df_to_vw
from Expert import Expert
from ExpertAdvice import ExpertAdvice
from constants import NUM_EXPERTS

df: pd.DataFrame; features: list[str]; training_examples: list[str]
df, features = read_df_from_file()
training_examples = convert_df_to_vw(df, features)

expert_learning_rates: list[float] = [0.1, 0.25, 0.5, 0.75, 1, 5]
experts: list[Expert] = [
    Expert(features=features, learning_rate=expert_learning_rates[i]) for i in range(NUM_EXPERTS)
]
learner_learning_rate: float = 0.5
learner: ExpertAdvice = ExpertAdvice(experts, alpha=0.3, learning_rate=learner_learning_rate)
for ex in training_examples:
    learner.learn(ex)

logs: list[str]; history: pd.DataFrame
history, expert_histories = learner.get_log()
for e_history in expert_histories:
    print(e_history[1].head())