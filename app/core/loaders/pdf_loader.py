from typing import Tuple, Dict
from pypdf import PdfReader
from io import BytesIO

from app.core.loaders.base import DocumentLoader


class PDFLoader(DocumentLoader):
    def load(self, file_bytes: bytes) -> Tuple[str, Dict]:
        reader = PdfReader(BytesIO(file_bytes))
        text = []
        metadata = {}
        for i, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text()
                if page_text:
                    text.append(page_text)
            except Exception as e:
                print(f"Skipping page {i}: {e}")

            page_metadata = {
                "extracted_length": len(page_text),
                "loader": "pdf",
            }
            metadata[f"page_{i+1}"] =  page_metadata

        return "\n".join(text), metadata