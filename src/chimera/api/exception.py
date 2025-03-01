from requests import Response  # type: ignore

from .response import get_error_response_message


class ResponseException(Exception):
    """
    Represents an exception that occurs due to an unsuccessful HTTP response.

    This exception class is designed to encapsulate a `requests.Response` object,
    allowing for detailed inspection of the failed request and its response.
    It provides a user-friendly string representation of the error by
    extracting and formatting the response message.
    """

    def __init__(self, response: Response, message: str = "") -> None:
        """
        Initializes the ResponseException with a given HTTP response.

        Args:
            response: The `requests.Response` object representing the HTTP response
                      that caused the exception.
        """
        self._response = response
        self._message = message

    def __str__(self) -> str:
        """
        Returns a string representation of the exception.

        This method delegates to the `get_error_response_message` function to format
        the underlying response object into a descriptive error message.

        Returns:
            A string describing the error based on the HTTP response.
        """
        try:
            return get_error_response_message(self._response) + "\n" + self._message
        except Exception:
            return self._message
