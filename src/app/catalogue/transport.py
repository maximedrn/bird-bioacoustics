"""Share HTTP failure classification across catalogue and audio requests."""

from http import HTTPStatus


def is_retryable_status(status: int) -> bool:
    """Identify server failures, request timeouts and rate limiting.

    :param status: HTTP response status.
    :type status: int
    :return: Whether another bounded request attempt is appropriate.
    :rtype: bool
    """
    return status >= HTTPStatus.INTERNAL_SERVER_ERROR or status in (
        HTTPStatus.REQUEST_TIMEOUT,
        HTTPStatus.TOO_MANY_REQUESTS,
    )
