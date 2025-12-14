"""
Pydantic models for API requests and responses.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request model for query endpoint."""

    question: str = Field(..., min_length=1, description="Question to ask the RAG system")
    n_results: Optional[int] = Field(None, ge=1, le=20, description="Number of chunks to retrieve")
    min_similarity: Optional[float] = Field(None, ge=0.0, le=1.0, description="Minimum similarity threshold")
    include_sources: bool = Field(True, description="Include source chunks in response")


class SourceInfo(BaseModel):
    """Information about a source chunk."""

    document: str = Field(..., description="Text content of the chunk")
    similarity: float = Field(..., description="Similarity score")
    source: str = Field(..., description="Source document name")


class QueryResponse(BaseModel):
    """Response model for query endpoint."""

    question: str = Field(..., description="Original question")
    answer: str = Field(..., description="Generated answer")
    num_sources: int = Field(..., description="Number of sources used")
    sources: Optional[List[SourceInfo]] = Field(None, description="Source chunks if requested")


class IngestionStats(BaseModel):
    """Statistics about document ingestion."""

    file_name: str = Field(..., description="Name of the ingested file")
    file_path: str = Field(..., description="Path to the file")
    num_characters: int = Field(..., description="Number of characters in document")
    num_chunks: int = Field(..., description="Number of chunks created")
    avg_chunk_size: int = Field(..., description="Average chunk size")
    status: str = Field(..., description="Ingestion status")


class IngestionResponse(BaseModel):
    """Response model for document ingestion."""

    message: str = Field(..., description="Status message")
    stats: IngestionStats = Field(..., description="Ingestion statistics")


class VectorStoreStats(BaseModel):
    """Statistics about the vector store."""

    collection_name: str = Field(..., description="Name of the collection")
    total_chunks: int = Field(..., description="Total number of chunks stored")
    unique_sources_sampled: int = Field(..., description="Number of unique source documents")
    persist_directory: str = Field(..., description="Persistence directory path")


class EmbeddingsInfo(BaseModel):
    """Information about the embeddings model."""

    model_name: str = Field(..., description="Name of the embedding model")
    embedding_dim: int = Field(..., description="Dimension of embeddings")
    device: str = Field(..., description="Device used for embeddings (cuda/cpu)")
    batch_size: int = Field(..., description="Batch size for processing")
    max_seq_length: int = Field(..., description="Maximum sequence length")


class SystemStats(BaseModel):
    """Complete system statistics."""

    vector_store: VectorStoreStats = Field(..., description="Vector store statistics")
    embeddings: EmbeddingsInfo = Field(..., description="Embeddings model information")
    llm_provider: str = Field(..., description="LLM provider (mistral/openai/anthropic)")
    llm_model: str = Field(..., description="LLM model name")
    chunk_size: int = Field(..., description="Chunk size for text splitting")
    chunk_overlap: int = Field(..., description="Overlap between chunks")
    retrieval_top_k: int = Field(..., description="Default number of chunks to retrieve")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    components: Dict[str, str] = Field(..., description="Status of each component")


class DeleteResponse(BaseModel):
    """Response for document deletion."""

    message: str = Field(..., description="Status message")
    source: str = Field(..., description="Deleted source identifier")


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
