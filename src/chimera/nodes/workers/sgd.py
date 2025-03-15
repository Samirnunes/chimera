from copy import deepcopy
from typing import Any, Literal, Type

import numpy as np
import pandas as pd
import uvicorn
from fastapi import APIRouter, FastAPI
from fastapi.responses import JSONResponse
from sklearn.linear_model import SGDClassifier, SGDRegressor

from ...api.configs import (
    CHIMERA_SGD_WORKER_FIT_REQUEST_DATA_SAMPLE_PATH,
    CHIMERA_SGD_WORKER_FIT_STEP_PATH,
)
from ...api.dto import FitStepInput, FitStepOutput, load_fit_input, load_fit_samples
from ...api.response import build_error_response, build_json_response
from ...containers.configs import (
    CHIMERA_TRAIN_DATA_FOLDER,
    CHIMERA_TRAIN_FEATURES_FILENAME,
    CHIMERA_TRAIN_LABELS_FILENAME,
    WorkersConfig,
)
from ...utils import status_logger

MODELS_MAP = {
    "regressor": SGDRegressor,
    "classifier": SGDClassifier,
}

MODEL_TYPE = Type[SGDRegressor | SGDClassifier]


class SGDWorker:
    """
    Implements a Stochastic Gradient Descent (SGD) worker node for Chimera.

    This class handles the training process for a single worker node using
    SGD, communicating with the master node to receive model parameters and
    return gradients.
    """

    def __init__(
        self,
        model: Literal["regressor", "classifier"],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Initializes the SGDWorker.

        Args:
            model: The type of model to use ("regressor" or "classifier").
            *args: Additional positional arguments passed to the model constructor.
            **kwargs: Additional keyword arguments passed to the model constructor.
        """
        self._model: MODEL_TYPE = MODELS_MAP[model](*args, **kwargs)
        self._weights: np.ndarray
        self._bias: float
        self._workers_config = WorkersConfig()
        self._partially_fitted = False
        self._X_train, self._y_train = load_fit_input(
            f"{CHIMERA_TRAIN_DATA_FOLDER}/{CHIMERA_TRAIN_FEATURES_FILENAME}",
            f"{CHIMERA_TRAIN_DATA_FOLDER}/{CHIMERA_TRAIN_LABELS_FILENAME}",
        )

    def serve(self) -> None:
        """
        Starts the FastAPI server for the model worker.
        """
        app = FastAPI()
        app.include_router(self._fit_router())
        status_logger.info(f"Serving {self.__class__.__name__}...")
        uvicorn.run(
            app,
            host=self._workers_config.CHIMERA_WORKERS_HOST,
            port=self._workers_config.CHIMERA_WORKERS_PORT,
        )

    def _receive_parameters(self) -> None:
        """
        (Not implemented)  Placeholder for receiving parameters from the master.
        """
        pass

    def _fit_router(self) -> APIRouter:
        """
        Creates and returns the FastAPI router for the /fit endpoint.

        Returns:
            The FastAPI router for fitting the model.
        """
        router = APIRouter()

        @router.get(CHIMERA_SGD_WORKER_FIT_REQUEST_DATA_SAMPLE_PATH)
        def request_data_sample() -> JSONResponse:
            """
            Returns a sample of the training data.
            """
            try:
                return build_json_response(
                    load_fit_samples(
                        f"{CHIMERA_TRAIN_DATA_FOLDER}/{CHIMERA_TRAIN_FEATURES_FILENAME}",
                        f"{CHIMERA_TRAIN_DATA_FOLDER}/{CHIMERA_TRAIN_LABELS_FILENAME}",
                    )
                )
            except Exception as e:
                status_logger.error(f"Error at {self.__class__.__name__}: {e}")
                return build_error_response(e)

        @router.post(CHIMERA_SGD_WORKER_FIT_STEP_PATH)
        def fit_step(fit_step_input: FitStepInput) -> JSONResponse:
            """
            Performs a single step of the SGD fitting process.
            """
            try:
                if not self._partially_fitted:
                    samples = load_fit_samples(
                        f"{CHIMERA_TRAIN_DATA_FOLDER}/{CHIMERA_TRAIN_FEATURES_FILENAME}",
                        f"{CHIMERA_TRAIN_DATA_FOLDER}/{CHIMERA_TRAIN_LABELS_FILENAME}",
                    )
                    y_train_samples = np.array(samples.y_train_sample_rows).ravel()
                    self._model.partial_fit(
                        pd.DataFrame(
                            samples.X_train_sample_rows,
                            columns=samples.X_train_sample_columns,
                        ),
                        y_train_samples,
                        classes=np.unique(y_train_samples),
                    )
                    self._partially_fitted = True
                else:
                    self._model.coef_ = np.array(fit_step_input.weights)
                    self._model.intercept_ = np.array(fit_step_input.bias)

                weights: np.ndarray = deepcopy(self._model.coef_)
                bias: np.ndarray = deepcopy(self._model.intercept_)

                self._model.partial_fit(self._X_train, self._y_train)

                weights_gradients: np.ndarray = weights - self._model.coef_
                bias_gradient: np.ndarray = bias - self._model.intercept_

                return build_json_response(
                    FitStepOutput(
                        weights_gradients=list(weights_gradients.flatten()),
                        bias_gradient=list(bias_gradient),
                    )
                )
            except Exception as e:
                status_logger.error(f"Error at {self.__class__.__name__}: {e}")
                return build_error_response(e)

        return router
