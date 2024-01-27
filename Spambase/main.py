import pandas as pd
import vowpalwabbit as vw
from vowpalwabbit.dftovw import DFtoVW, SimpleLabel, Feature
import random

DATA_PATH = "dataset/spambase.data"
NAMES_PATH = "dataset/spambase.names"
LABEL_COL_NAME = "is_spam"
df: pd.DataFrame = pd.read_csv(DATA_PATH, header=None)
with open(NAMES_PATH, 'r') as f:
    features_unparsed: list[str] = f.readlines()[1-len(df.columns)::]
features: list[str] = [column[:column.index(':'):] for column in features_unparsed]
columns: list[str] = features + [LABEL_COL_NAME]
df.columns = columns

converter: DFtoVW = DFtoVW(
    df=df,
    features=[Feature(feature) for feature in features],
    label=SimpleLabel(LABEL_COL_NAME)
)
training_examples: list[str] = converter.convert_df()
random.shuffle(training_examples)

model = vw.Workspace(P=1, enable_logging=True)

for ex in training_examples:
    model.learn(ex)

model.finish()