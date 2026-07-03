from abc import ABC, abstractmethod
from app.core.documents.elements import ExtractedDocument


class DocumentLoader(ABC):
    @abstractmethod
    def load(self, file_bytes: bytes) -> ExtractedDocument:
        """
        Returns:
        - extracted_text
        - metadata (extraction stats, warnings)
        """
        pass
