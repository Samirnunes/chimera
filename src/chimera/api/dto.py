from typing import List

from pydantic import BaseModel, field_validator
from pydantic_settings import NoDecode
from typing_extensions import Annotated

serializable = str | int | float | bool


class FitInput(BaseModel):
    X_train_columns: Annotated[List[str], NoDecode]
    X_train_rows: List[List[serializable]]
    y_train_columns: Annotated[List[str], NoDecode]
    y_train_rows: List[List[serializable]]

    @field_validator("X_train_columns", "y_train_columns", mode="before")
    @classmethod
    def normalize_columns(cls, columns: List[str]) -> List[str]:
        return sorted([column.lower().strip() for column in columns])


class PredictInput(BaseModel):
    X_pred_columns: Annotated[List[str], NoDecode]
    X_pred_rows: List[List[serializable]]

    @field_validator("X_pred_columns", mode="before")
    @classmethod
    def normalize_columns(cls, columns: List[str]) -> List[str]:
        return sorted([column.lower().strip() for column in columns])


class PredictOutput(BaseModel):
    y_pred_rows: List[serializable]


class FitOutput(BaseModel):
    fit: str = "ok"
