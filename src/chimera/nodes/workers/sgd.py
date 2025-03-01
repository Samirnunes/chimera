from sklearn.linear_model import LinearRegression, LogisticRegression

MODELS_MAP = {
    "linear_regression": LinearRegression(),
    "logistic_regression": LogisticRegression(),
}

MODEL_TYPE = LinearRegression | LogisticRegression


class SGDWorker:
    pass
