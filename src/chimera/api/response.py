from fastapi.responses import JSONResponse
from pydantic import BaseModel
from requests import Response  # type: ignore


def build_json_response(model: BaseModel) -> JSONResponse:
    """
    Builds a JSON response from a Pydantic model.

    Args:
        model: The Pydantic model to serialize into JSON.

    Returns:
        A FastAPI JSONResponse object with the serialized model data and a 200 status code.
    """
    return JSONResponse(model.model_dump(), 200)


def build_error_response(
    exception: Exception, status_code: int = 500
) -> JSONResponse:
    """
    Builds a JSON response for an error.

    Args:
        exception: The exception that occurred.
        status_code: The HTTP status code for the error response (default: 500).

    Returns:
        A FastAPI JSONResponse object containing error details and the specified status code.
    """
    return JSONResponse(
        content={
            "message": str(exception),
            "details": {
                "errorCode": status_code,
                "errorMessage": exception.__class__.__name__,
            },
        },
        status_code=status_code,
    )


def get_error_response_message(response: Response) -> str:
    """
    Extracts the 'message' field from a JSON response.

    Assumes the response body is a JSON object with a 'message' key.

    Args:
        response: The requests.Response object containing the JSON response.

    Returns:
        The value of the 'message' field as a string.
    """
    return response.json()["message"]
