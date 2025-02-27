from requests import Response  # type: ignore

from .response import get_response_message


class ResponseException(Exception):
    def __init__(self, response: Response) -> None:
        self._response = response

    def __str__(self) -> str:
        return get_response_message(self._response)
