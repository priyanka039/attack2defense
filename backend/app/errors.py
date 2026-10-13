"""One error shape for every failure: ``{"error": {"code", "message", "field"}}``."""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger("attack2defense.api")

HTTP_CODES = {
    400: "bad_request",
    401: "not_authenticated",
    403: "forbidden",
    404: "not_found",
    405: "method_not_allowed",
    409: "conflict",
    422: "invalid_input",
    503: "unavailable",
}


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str, field: str | None = None):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.field = field


def not_found(what: str, identifier: str) -> ApiError:
    return ApiError(404, f"{what}_not_found", f"No {what.replace('_', ' ')} with id '{identifier}'.")


def _body(code: str, message: str, field: str | None = None) -> dict:
    return {"error": {"code": code, "message": message, "field": field}}


def _validation_message(error: dict) -> tuple[str, str | None]:
    location = [str(part) for part in error.get("loc", ()) if part not in ("body", "query", "path")]
    field = ".".join(location) or None
    message = str(error.get("msg", "Invalid value"))
    message = message.removeprefix("Value error, ")
    if error.get("type") == "missing":
        message = "This field is required"
    return (f"{field}: {message}" if field else message), field


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def api_error(_: Request, exc: ApiError):
        return JSONResponse(_body(exc.code, exc.message, exc.field), status_code=exc.status)

    @app.exception_handler(RequestValidationError)
    async def validation_error(_: Request, exc: RequestValidationError):
        errors = exc.errors()
        message, field = _validation_message(errors[0]) if errors else ("Invalid request", None)
        return JSONResponse(_body("invalid_input", message, field), status_code=422)

    @app.exception_handler(StarletteHTTPException)
    async def http_error(_: Request, exc: StarletteHTTPException):
        code = HTTP_CODES.get(exc.status_code, "error")
        message = exc.detail if isinstance(exc.detail, str) else "Request failed"
        return JSONResponse(_body(code, message), status_code=exc.status_code, headers=exc.headers)

    @app.exception_handler(Exception)
    async def unexpected_error(_: Request, exc: Exception):
        logger.exception("Unhandled error", exc_info=exc)
        return JSONResponse(
            _body("server_error", "Something went wrong on the server. Check the backend log."),
            status_code=500,
        )
