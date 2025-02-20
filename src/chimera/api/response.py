from fastapi.responses import JSONResponse
from pydantic import BaseModel


def build_json_response(model: BaseModel) -> JSONResponse:
    return JSONResponse(model.__dict__, 200)


def build_error_response(
    exception: Exception, status_code: int = 500
) -> JSONResponse:
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
