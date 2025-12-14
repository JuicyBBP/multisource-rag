"""
API routes for MultiSource RAG System.
"""

from pathlib import Path
from typing import List
from fastapi import APIRouter, UploadFile, File, HTTPException, status
from loguru import logger

from src.config import settings
from src.api.models import (
    QueryRequest,
    QueryResponse,
    IngestionResponse,
    IngestionStats,
    SystemStats,
    HealthResponse,
    DeleteResponse,
    ErrorResponse,
    SourceInfo,
)
from src.services.rag_service import RAGService


# Create router
router = APIRouter()

# Initialize RAG service (singleton)
rag_service: RAGService | None = None


def get_rag_service() -> RAGService:
    """Get or create RAG service instance."""
    global rag_service
    if rag_service is None:
        logger.info("Initializing RAG service...")
        rag_service = RAGService()
    return rag_service


@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint."""
    try:
        rag = get_rag_service()

        components = {
            "rag_service": "operational",
            "vector_store": "operational",
            "embeddings": "operational",
            "llm": "operational",
        }

        return HealthResponse(
            status="healthy",
            version=settings.app_version,
            components=components,
        )

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"error": "HealthCheckFailed", "message": str(e)},
        )


@router.get("/stats", response_model=SystemStats, tags=["System"])
async def get_stats():
    """Get system statistics."""
    try:
        rag = get_rag_service()
        stats = rag.get_stats()
        return SystemStats(**stats)

    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "StatsError", "message": str(e)},
        )


@router.post("/ingest", response_model=IngestionResponse, tags=["Documents"])
async def ingest_document(file: UploadFile = File(...)):
    """Ingest a document into the RAG system.

    Accepts: PDF, DOCX, TXT, MD files
    """
    try:
        # Validate file type
        allowed_extensions = {".pdf", ".docx", ".doc", ".txt", ".md", ".markdown"}
        file_ext = Path(file.filename).suffix.lower()

        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "InvalidFileType",
                    "message": f"File type {file_ext} not supported. Allowed: {', '.join(allowed_extensions)}",
                },
            )

        # Check file size
        if file.size and file.size > settings.max_upload_size_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail={
                    "error": "FileTooLarge",
                    "message": f"File size exceeds {settings.max_upload_size_mb}MB limit",
                },
            )

        # Save uploaded file
        settings.upload_directory.mkdir(parents=True, exist_ok=True)
        file_path = settings.upload_directory / file.filename

        # Write file
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        logger.info(f"Saved uploaded file: {file_path}")

        # Ingest document
        rag = get_rag_service()
        stats = rag.ingest_document(file_path, show_progress=False)

        return IngestionResponse(
            message=f"Document '{file.filename}' ingested successfully",
            stats=IngestionStats(**stats),
        )

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"Error ingesting document: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "IngestionError", "message": str(e)},
        )


@router.post("/query", response_model=QueryResponse, tags=["Query"])
async def query(request: QueryRequest):
    """Query the RAG system with a question."""
    try:
        rag = get_rag_service()

        response = rag.query(
            question=request.question,
            n_results=request.n_results,
            min_similarity=request.min_similarity,
            include_sources=request.include_sources,
        )

        # Convert sources to SourceInfo objects if present
        if response.get("sources"):
            response["sources"] = [
                SourceInfo(**source) for source in response["sources"]
            ]

        return QueryResponse(**response)

    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "QueryError", "message": str(e)},
        )


@router.delete("/documents/{source}", response_model=DeleteResponse, tags=["Documents"])
async def delete_document(source: str):
    """Delete all chunks from a specific document."""
    try:
        rag = get_rag_service()
        rag.delete_document(source)

        return DeleteResponse(
            message=f"Document '{source}' deleted successfully",
            source=source,
        )

    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "DeleteError", "message": str(e)},
        )


@router.post("/reset", tags=["System"])
async def reset_system():
    """Reset the RAG system (delete all documents).

    WARNING: This will delete all stored documents and cannot be undone!
    """
    try:
        rag = get_rag_service()
        rag.reset()

        return {
            "message": "RAG system reset successfully",
            "warning": "All documents have been deleted",
        }

    except Exception as e:
        logger.error(f"Error resetting system: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "ResetError", "message": str(e)},
        )
