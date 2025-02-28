from requests import Response  # type: ignore

from .response import get_response_message


class ResponseException(Exception):
    """
    Represents an exception that occurs due to an unsuccessful HTTP response.

    This exception class is designed to encapsulate a `requests.Response` object,
    allowing for detailed inspection of the failed request and its response.
    It provides a user-friendly string representation of the error by
    extracting and formatting the response message.
    """

    def __init__(self, response: Response) -> None:
        """
        Initializes the ResponseException with a given HTTP response.

        Args:
            response: The `requests.Response` object representing the HTTP response
                      that caused the exception.
        """
        self._response = response

    def __str__(self) -> str:
        """
        Returns a string representation of the exception.

        This method delegates to the `get_response_message` function to format
        the underlying response object into a descriptive error message.

        Returns:
            A string describing the error based on the HTTP response.
        """
        return get_response_message(self._response)
