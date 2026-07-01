from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from app.core.errors import AppError
from app.core.logging import setup_logging
from app.core.config import Settings
from app.core.request_id import request_id_middleware
from app.core.observability import observability_middleware
from app.core.timeouts import TimeoutMiddleware
from app.api import health, ingest, query, metrics

_settings = Settings()
setup_logging(log_level=_settings.log_level)

app = FastAPI(title="Applied AI System")

@app.exception_handler(AppError)
async def app_error_handler(request, exc: AppError):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.code,
            "message": exc.message,
            "request_id": request.state.request_id,
        },
    )

app.middleware("http")(TimeoutMiddleware(timeout_seconds=30))
app.middleware("http")(observability_middleware)
app.middleware("http")(request_id_middleware)

app.include_router(health.router)
app.include_router(ingest.router)
app.include_router(query.router)
app.include_router(metrics.router)