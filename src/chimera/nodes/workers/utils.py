import pandas as pd

from ...api.dto import FitInput


def load_csv_as_fit_input(x_train_path: str, y_train_path: str) -> FitInput:
    X_train_df = pd.read_csv(x_train_path)
    y_train_df = pd.read_csv(y_train_path)

    return FitInput(
        X_train_columns=list(X_train_df.columns),
        X_train_rows=X_train_df.astype(object).values.tolist(),
        y_train_columns=list(y_train_df.columns),
        y_train_rows=y_train_df.astype(object).values.tolist(),
    )
