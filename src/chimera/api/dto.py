from shutil import ReadError
from typing import List, Tuple

import pandas as pd
from pydantic import BaseModel, field_validator
from pydantic_settings import NoDecode
from typing_extensions import Annotated

serializable = str | int | float | bool


def load_fit_input(
    x_train_path: str, y_train_path: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads training data from CSV files.

    Args:
        x_train_path: Path to the CSV file containing training features (X).
        y_train_path: Path to the CSV file containing training labels (y).
    """
    X_train = pd.read_csv(x_train_path)
    y_train = pd.read_csv(y_train_path)

    def normalize_columns(columns: List[str]) -> List[str]:
        """Normalizes column names to lowercase and removes whitespace."""
        return [column.lower().strip() for column in columns]

    X_train.columns = normalize_columns(X_train.columns)

    return X_train, y_train


class FitOutput(BaseModel):
    """
    Data transfer object (DTO) for the fit operation output.
    """

    fit: str = "ok"
    """A simple confirmation message indicating successful fit."""


class FitStepInput(BaseModel):
    """
    Data transfer object (DTO) for a single step in the fitting process.  Contains weights and bias.
    """

    weights: List[float]
    """List of weights for the model."""
    bias: List[float]
    """List of bias terms for the model."""


class FitStepOutput(BaseModel):
    """
    Data transfer object (DTO) for the output of a single fit step. Contains gradients.
    """

    weights_gradients: List[float]
    """List of gradients for the weights."""
    bias_gradient: List[float]
    """Gradient for the bias term."""


class FitRequestDataSampleOutput(BaseModel):
    """
    Data transfer object (DTO) for a sample of the training data used in a fit request.
    """

    X_train_sample_columns: Annotated[List[str], NoDecode]
    """List of column names for the sample of training features (X)."""
    X_train_sample_rows: List[List[serializable]]
    """List of rows for the sample of training features (X). Each row is a list of serializable values."""
    y_train_sample_columns: Annotated[List[str], NoDecode]
    """List of column names for the sample of training labels (y)."""
    y_train_sample_rows: List[List[serializable]]
    """List of rows for the sample of training labels (y). Each row is a list of serializable values."""

    @field_validator(
        "X_train_sample_columns", "y_train_sample_columns", mode="before"
    )
    @classmethod
    def normalize_columns(cls, columns: List[str]) -> List[str]:
        """Normalizes column names to lowercase and removes whitespace."""
        return [column.lower().strip() for column in columns]


class PredictInput(BaseModel):
    """
    Data transfer object (DTO) for the predict operation. Contains prediction data.
    """

    X_pred_columns: Annotated[List[str], NoDecode]
    """List of column names for the prediction features (X)."""
    X_pred_rows: List[List[serializable]]
    """List of rows for the prediction features (X). Each row is a list of serializable values."""

    @field_validator("X_pred_columns", mode="before")
    @classmethod
    def normalize_columns(cls, columns: List[str]) -> List[str]:
        """Normalizes column names to lowercase and removes whitespace."""
        return [column.lower().strip() for column in columns]


class PredictOutput(BaseModel):
    """
    Data transfer object (DTO) for the predict operation output.
    """

    y_pred_rows: List[serializable]
    """List of predicted values."""


def load_csv_sample(
    x_train_path: str, y_train_path: str
) -> FitRequestDataSampleOutput:
    """
    Loads a sample of training data from CSV files and converts it into a
    FitRequestDataSampleOutput DTO.  This function attempts to load progressively
    smaller samples of the data until a successful load occurs.

    Args:
        x_train_path: Path to the CSV file containing training features (X).
        y_train_path: Path to the CSV file containing training labels (y).

    Returns:
        A FitRequestDataSampleOutput DTO containing a sample of the training data.

    Raises:
        ReadError: If all attempts to load a sample of the specified size fail.  This indicates a problem with the input CSV files.
    """

    rows = [200, 100, 50, 25, 10, 5, 2]

    for row in rows:
        try:
            X_train_sample = pd.read_csv(x_train_path, nrows=row)
            y_train_sample = pd.read_csv(y_train_path, nrows=row)
            break
        except Exception:
            if row == rows[-1]:
                raise ReadError()
            continue

    return FitRequestDataSampleOutput(
        X_train_sample_columns=list(X_train_sample.columns),
        X_train_sample_rows=list(X_train_sample.values),
        y_train_sample_columns=list(y_train_sample.columns),
        y_train_sample_rows=list(y_train_sample.values),
    )
