"""
Code to read in data and convert it to Vowpal Wabbit-readable format.
"""
import pandas as pd
from vowpalwabbit.dftovw import DFtoVW, Feature, SimpleLabel
from constants import SPAMBASE, CLOUD, SPAMBASE_DATA_PATH, SPAMBASE_NAMES_PATH, \
    SPAMBASE_LABEL_COL_NAME, CLOUD_DATA_PATH, CLOUD_FEATURES, CLOUD_LABEL_COL_NAME
from random import shuffle
import re

def load_data(dataset: str = SPAMBASE) -> tuple[pd.DataFrame, list[str]]:
    label_name: str; df: pd.DataFrame; features: list[str]
    if dataset == SPAMBASE:
        label_name = SPAMBASE_LABEL_COL_NAME
        df = pd.read_csv(SPAMBASE_DATA_PATH, header=None)
        with open(SPAMBASE_NAMES_PATH, 'r') as f:
            features_unparsed: list[str] = f.readlines()[1-len(df.columns)::]
        features = [column[:column.index(':'):] for column in features_unparsed]
        df.columns = features + [label_name]
    elif dataset == CLOUD:
        label_name = CLOUD_LABEL_COL_NAME
        features = CLOUD_FEATURES
        df = pd.DataFrame(columns=CLOUD_FEATURES + [label_name])
        reading_numbers: bool = False
        label: int = -1
        with open(CLOUD_DATA_PATH, 'r') as f:
            lines: list[str] = f.readlines()
            for line in lines:
                if re.match(r"^[\d|\.|-]+$", "".join(line.split())):
                    if not reading_numbers:
                        label += 1
                        reading_numbers = True
                    df_row: list[any] = [
                        float(entry) for entry in re.findall(r"[\d|\.|-]+", line)
                    ] + [label]
                    df.loc[len(df)] = df_row
                else:
                    reading_numbers = False
    else:
        raise Exception(f"Invalid dataset {dataset}")
    df[label_name] = (df[label_name] * 2 - 1).astype(int)
    return df, features


def read_df_from_file(
    data_file: str,
    names_file: str,
    label_name: str,
    features: list[str]
) -> tuple[pd.DataFrame, list[str]]:
    """
    Reads in a .data and .names file into a dataframe and separates out the 
    label into the last column.
    """
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
    label_name: str = SPAMBASE_LABEL_COL_NAME
) -> list[str]:
    """
    Converts the DataFrame to Vowpal Wabbit format for learning by Vowpal Wabbit models.
    """
    converter: DFtoVW = DFtoVW(
        df=df,
        features=[Feature(feature) for feature in features],
        label=SimpleLabel(label_name)
    )
    training_examples: list[str] = converter.convert_df()
    shuffle(training_examples)
    return training_examples