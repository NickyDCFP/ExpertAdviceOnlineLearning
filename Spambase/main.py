import pandas as pd
from fileio import read_df_from_file, convert_df_to_vw
from Expert import Expert

df: pd.DataFrame; features: list[str]; training_examples: list[str]
df, features = read_df_from_file()
training_examples = convert_df_to_vw(df, features)

learning_rate: float = 0.5
loss: str = "hinge"
expert: Expert = Expert(features=features, learning_rate=learning_rate, loss_function=loss)
for ex in training_examples:
    expert.learn(ex)

logs: list[str]; history: pd.DataFrame
logs, history = expert.get_log()