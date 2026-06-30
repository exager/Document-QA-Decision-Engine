import logging
from fastapi import APIRouter, HTTPException, Request
from app.schemas.ingest import IngestRequest, IngestResponse
from app.core.retry import retry
from app.core.limits import enforce_size_limit
from app.core.search.dummy_internet import DummyInternetSearchProvider
import uuid
import hashlib
import json
import numpy as np
from typing import List
from app.core.documents.models import Document
from app.core.documents.chunker import chunk_document
from fastapi import UploadFile, File
from app.core.loaders.pdf_loader import PDFLoader
from app.core.loaders.docx_loader import DocxLoader
from app.core.loaders.quality import validate_extraction
from app.core.state import state
from app.core.storage.vector_index import InMemoryVectorIndex
from app.core.process import process_document


router = APIRouter()
logger = logging.getLogger(__name__)

search_provider = DummyInternetSearchProvider()

@router.post("/ingest", response_model=IngestResponse)
async def ingest(req: IngestRequest, request: Request):
    await enforce_size_limit(request)
    request_id = request.state.request_id

    documents: List[Document] = []

    if req.source == "internet":
        if not req.search_query:
            raise HTTPException(
                status_code=400,
                detail="search_query is required when source=internet"
            )

        results = retry(
            lambda:search_provider.search(req.search_query),
            retries = 3,
        )
        # discovered_docs = len(results)
        for result in results:
            # Stable ID derived from source URL
            source_doc_id = hashlib.sha256(f'url:{result.url}'.encode("utf-8")).hexdigest()

            documents.append(
                Document(
                    document_id=source_doc_id,
                    content=result.content,
                    metadata={
                        "source": "internet",
                        "url": result.url,
                        "title": result.title,
                    },
                )
            )
        logger.info(
            "internet_documents_discovered",
            extra={
                "request_id": request_id,
                "query": req.search_query,
            }
        )

    elif req.source == "local":
        if not req.content:
            raise HTTPException(
                status_code=400,
                detail="content is required when source=local"
            )

        local_doc_id = hashlib.sha256(f'local:{req.content}'.encode("utf-8")).hexdigest()
        documents.append(
            Document(
                document_id=local_doc_id,
                content=req.content,
                metadata={
                    "source": "local",
                    **(req.metadata or {}),
                },
            )
        )
        logger.info(
            "local_document_received",
            extra={
                "request_id": request_id,
                "document_id": local_doc_id,
            },
        )
    else:
        raise HTTPException(status_code=400, detail="invalid source")

    #Process Documents
    processed_count = process_document(documents, request_id)
    return IngestResponse(
        document_id="batch",
        status="accepted",
        discovered_docs=len(documents),
    )

@router.post("/ingest/file", response_model=IngestResponse)
async def ingest_file(
    request: Request = None,
    file: UploadFile = File(...),
):
    request_id = request.state.request_id

    pdf_loader = PDFLoader()
    docx_loader = DocxLoader()
    file_bytes = await file.read()

    if file.filename.endswith(".pdf"):
        text, meta = pdf_loader.load(file_bytes)
    elif file.filename.endswith(".docx"):
        text, meta = docx_loader.load(file_bytes)
    else:
        raise HTTPException(status_code=400, detail="unsupported_file_type")

    try:
        validate_extraction(text)
    except ValueError:
        logger.warning(
            "document_extraction_refused",
            extra={
                "request_id": request_id,
                "filename": file.filename,
                "extracted_length": meta["extracted_length"],
            },
        )
        raise HTTPException(status_code=422, detail="low_quality_document")

    document_id = str(uuid.uuid4())

    document = Document(
        document_id=document_id,
        content=text,
        metadata={
            "source": "file",
            "filename": file.filename,
            **meta,
        },
    )

    processed_docs = process_document([document], request_id)

    logger.info(
        "file_document_ingested",
        extra={
            "request_id": request_id,
            "document_id": document_id,
            "newly_embedded_chunks": processed_docs,
        },
    )

    return IngestResponse(
        document_id=document_id,
        status="accepted",
        discovered_docs=1,
    )
