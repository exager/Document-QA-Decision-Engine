from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from app.core.errors import AppError, InternalError
from app.core.logging import setup_logging
from app.core.config import get_app_settings
from app.core.request_id import request_id_middleware
from app.core.observability import observability_middleware
from app.core.timeouts import TimeoutMiddleware
from app.api import health, ingest, query, metrics

_settings = get_app_settings()
setup_logging(log_level=_settings.log_level)

app = FastAPI(title="Applied AI System")

# Domain error handler for base AppErrors
@app.exception_handler(AppError)
async def app_error_handler(request, exc: AppError):
    request_id = getattr(request.state, "request_id", "unknown")
    return JSONResponse(
        status_code=exc.status_code,
        content=exc.to_dict(request_id),
    )

# For the cases FastAPI raises an HTTPException
# Not used now, bbut may be needed later
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """
    Override FastAPI's default HTTPException response so error bodies
    include the request_id. Same shape as AppError responses, so clients
    only have to know one error envelope.
    """
    # request_id_middleware sets this at the start of every request. Use
    # getattr rather than direct access because a handler can fire before
    # the middleware ran (extremely rare, but possible on middleware setup
    # errors) — better to return "unknown" than crash inside the handler.
    request_id = getattr(request.state, "request_id", "unknown")

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "http_exception",
            "message": exc.detail,
            "request_id": request_id,
        },
        headers=exc.headers,   # preserve WWW-Authenticate etc. if set
    )


# Pydantic validation error handler: same envelope, distinct code.
@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    request_id = getattr(request.state, "request_id", "unknown")
    return JSONResponse(
        status_code=422,
        content={
            "error": "validation_error",
            "message": "request failed validation",
            "request_id": request_id,
            "details": {"errors": exc.errors()},
        },
    )

# Catch-all for anything else. Never leaks stack traces to clients.
@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    request_id = getattr(request.state, "request_id", "unknown")
    # Log the full traceback server-side; return an opaque envelope client-side.
    import logging
    logging.getLogger(__name__).exception(
        "unhandled_exception",
        extra={"request_id": request_id, "path": str(request.url)},
    )
    err = InternalError("internal error")
    return JSONResponse(
        status_code=err.status_code,
        content=err.to_dict(request_id),
    )


# Ordering matters, fastapi executes middleware in reverse order
# Firstly outermost is set, request_id..., then ocservability
# And finally timeout.
app.middleware("http")(TimeoutMiddleware(timeout_seconds=_settings.request_timeout_seconds))
app.middleware("http")(observability_middleware)
app.middleware("http")(request_id_middleware)

app.include_router(health.router)
app.include_router(ingest.router)
app.include_router(query.router)
app.include_router(metrics.router)