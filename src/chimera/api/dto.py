from typing import Dict, List

from pydantic import BaseModel


class PredictInput(BaseModel):
    X: Dict


class PredictOutput(BaseModel):
    y_pred: List


class FitOutput(BaseModel):
    fit: str = "ok"
