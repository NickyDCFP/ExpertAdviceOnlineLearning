import pandas as pd
from vowpalwabbit.dftovw import DFtoVW, Feature, SimpleLabel
from constants import DATA_PATH, NAMES_PATH, LABEL_COL_NAME
from random import shuffle

def read_df_from_file(
    data_file: str = DATA_PATH,
    names_file: str = NAMES_PATH,
    label_name: str = LABEL_COL_NAME
) -> tuple[pd.DataFrame, list[str]]:
    df: pd.DataFrame = pd.read_csv(data_file, header=None)
    with open(names_file, 'r') as f:
        features_unparsed: list[str] = f.readlines()[1-len(df.columns)::]
    features: list[str] = [column[:column.index(':'):] for column in features_unparsed]
    columns: list[str] = features + [label_name]
    df.columns = columns
    df[label_name] = df[label_name] * 2 - 1
    return df, features

def convert_df_to_vw(
    df: pd.DataFrame,
    features: list[str],
    label_name: str = LABEL_COL_NAME
) -> list[str]:
    converter: DFtoVW = DFtoVW(
        df=df,
        features=[Feature(feature) for feature in features],
        label=SimpleLabel(label_name)
    )
    training_examples: list[str] = converter.convert_df()
    shuffle(training_examples)
    return training_examples