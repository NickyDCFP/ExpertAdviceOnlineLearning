SPAMBASE: str = "SPAMBASE"
CLOUD: str = "CLOUD"
# The path to the Spambase dataset
SPAMBASE_DATA_PATH: str = "dataset/spambase.data"
# The path to the Spambase dataset's .names file
SPAMBASE_NAMES_PATH: str = "dataset/spambase.names"
# The column name for the label in Spambase
SPAMBASE_LABEL_COL_NAME: str = "is_spam"
# The path to the cloud dataset
CLOUD_DATA_PATH: str = "dataset/cloud.data"
# The column name for the label in the Cloud dataset
CLOUD_LABEL_COL_NAME: str = "cover_db"
# The features for the Cloud dataset
CLOUD_FEATURES: list[str] = [
    "visible_mean",
    "visible_max",
    "visible_min",
    "visible_mean_dist",
    "visible_contrast",
    "visible_entropy",
    "visible_momentum",
    "IR_mean",
    "IR_max",
    "IR_min"
]
# The metrics on the model to be collected at each timestep.
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
# The number of experts for the Experts Advice algorithm to host.
NUM_EXPERTS = 6