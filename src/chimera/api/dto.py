from typing import Any, List

import pandas as pd
from pydantic import BaseModel, field_validator
from pydantic_settings import NoDecode
from typing_extensions import Annotated

serializable = str | int | float | bool


class FitInput(BaseModel):
    """
    Data transfer object (DTO) for the fit operation. Contains training data.
    """

    X_train_columns: Annotated[List[str], NoDecode]
    """List of column names for the training features (X)."""
    X_train_rows: List[List[serializable]]
    """List of rows for the training features (X). Each row is a list of serializable values."""
    y_train_columns: Annotated[List[str], NoDecode]
    """List of column names for the training labels (y)."""
    y_train_rows: List[List[serializable]]
    """List of rows for the training labels (y). Each row is a list of serializable values."""

    def model_post_init(self, __context: Any) -> None:
        if len(self.X_train_rows) != len(self.y_train_rows):
            raise ValueError(
                "X_train_rows and y_train_rows must have the same length"
            )

    @field_validator("X_train_columns", "y_train_columns", mode="before")
    @classmethod
    def normalize_columns(cls, columns: List[str]) -> List[str]:
        """Normalizes column names to lowercase and removes whitespace."""
        return sorted([column.lower().strip() for column in columns])


class FitOutput(BaseModel):
    """
    Data transfer object (DTO) for the fit operation output.
    """

    fit: str = "ok"
    """A simple confirmation message indicating successful fit."""


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
        return sorted([column.lower().strip() for column in columns])


class PredictOutput(BaseModel):
    """
    Data transfer object (DTO) for the predict operation output.
    """

    y_pred_rows: List[serializable]
    """List of predicted values."""


def load_csv_as_fit_input(x_train_path: str, y_train_path: str) -> FitInput:
    """
    Loads training data from CSV files and converts it into a FitInput DTO.

    Args:
        x_train_path: Path to the CSV file containing training features (X).
        y_train_path: Path to the CSV file containing training labels (y).

    Returns:
        A FitInput DTO containing the loaded training data.
    """
    X_train_df = pd.read_csv(x_train_path)
    y_train_df = pd.read_csv(y_train_path)

    return FitInput(
        X_train_columns=list(X_train_df.columns),
        X_train_rows=X_train_df.astype(object).values.tolist(),
        y_train_columns=list(y_train_df.columns),
        y_train_rows=y_train_df.astype(object).values.tolist(),
    )
