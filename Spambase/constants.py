DATA_PATH: str = "dataset/spambase.data"
NAMES_PATH: str = "dataset/spambase.names"
LABEL_COL_NAME: str = "is_spam"
EXPERT_METRICS: list[str] = [
    "average loss",
    "sample loss",
    "sample number",
    "cumulative sample weight",
    "true label",
    "predicted label",
    "number of features"
]