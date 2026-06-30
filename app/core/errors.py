class AppError(Exception):
    code = "app_error"
    status_code = 500

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class IngestionError(AppError):
    code = "ingestion_failed"
    status_code = 422


class ExternalDependencyError(AppError):
    code = "external_dependency_failed"
    status_code = 503
