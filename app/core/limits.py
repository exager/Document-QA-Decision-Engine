from fastapi import Request, UploadFile
from app.core.config import get_app_settings
from app.core.errors import PayloadTooLargeError


_settings = get_app_settings()
MAX_JSON_BODY_BYTES = _settings.max_json_body_bytes       # (~100 KB)
MAX_FILE_UPLOAD_BYTES = _settings.max_file_upload_bytes   # (~25 MB)

async def enforce_json_size_limit(request: Request, max_bytes: int) -> None:
    """
    Size guard for JSON / form-URL-encoded routes. Reads the body via
    request.body(). MUST NOT be used on routes that also consume UploadFile,
    because request.body() and UploadFile share the same underlying stream.
    """
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            declared_size = int(content_length)
        except ValueError:
            declared_size = None

        if declared_size is not None and declared_size > max_bytes:
            raise PayloadTooLargeError(
                "payload too large",
                details={
                    "limit_bytes": max_bytes,
                    "payload_size": declared_size,
                },
            )
        
    body = await request.body()
    if len(body) > max_bytes:
        raise PayloadTooLargeError(
            "payload too large",
            details={
                "limit_bytes": max_bytes,
                "payload_size": len(body),
            },
        )


async def enforce_upload_size_limit(
    request: Request,
    file: UploadFile,
    max_bytes: int = MAX_FILE_UPLOAD_BYTES,
) -> None:
    """
    Size guard for multipart file uploads. Uses:
      1) Content-Length header (cheap, catches most cases before we buffer).
      2) The UploadFile's spooled temp file size, measured without consuming
         the stream (seek to end, tell, seek back).
    Note: Can't use await on request.body()
    """
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            if int(content_length) > max_bytes:
                raise PayloadTooLargeError(
                    "payload too large",
                    details={
                        "limit_bytes": max_bytes,
                        "payload_size": int(content_length),
                    },
                )
        except ValueError:
            pass

    try:
        await file.seek(0, 2)
        size = file.file.tell()
        await file.seek(0)
    except Exception:
        return

    if size > max_bytes:
        raise PayloadTooLargeError(
            "payload too large",
            details={
                "limit_bytes": max_bytes,
                "payload_size": size,
            },
        )