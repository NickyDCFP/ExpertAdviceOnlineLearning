DATA_PATH: str = "dataset/spambase.data"
NAMES_PATH: str = "dataset/spambase.names"
LABEL_COL_NAME: str = "is_spam"
METRICS: list[str] = [
    "true label",
    "predicted label",
    "cumulative loss",
    "average loss",
    "sample loss",
]
# Index cutoff for label in example strings.
#   Eg, example is formatted as "1 | features" or "-1 | features", so we can find the label in the
#   first two characters of the example.
LABEL_FORMAT_INDEX: int = 2
LOGISTIC: str = "logistic"
NUM_EXPERTS = 6