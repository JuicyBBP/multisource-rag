"""
Configuration module for MultiSource RAG System.
Loads and validates environment variables using Pydantic Settings.
"""

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.absolute()


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ========================================
    # LLM Configuration
    # ========================================
    mistral_api_key: str = Field(..., description="Mistral API key")
    openai_api_key: str | None = Field(None, description="OpenAI API key (optional)")
    anthropic_api_key: str | None = Field(None, description="Anthropic API key (optional)")

    llm_provider: Literal["mistral", "openai", "anthropic"] = Field(
        default="mistral", description="LLM provider to use"
    )
    llm_model: str = Field(
        default="mistral-small-latest", description="LLM model name"
    )
    llm_temperature: float = Field(
        default=0.1, ge=0.0, le=2.0, description="LLM temperature"
    )
    llm_max_tokens: int = Field(
        default=1500, ge=100, le=4000, description="Max tokens for LLM response"
    )

    # ========================================
    # Embeddings Configuration
    # ========================================
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Sentence transformer model for embeddings",
    )
    embedding_device: Literal["cuda", "cpu"] = Field(
        default="cuda", description="Device for embeddings (cuda or cpu)"
    )
    embedding_batch_size: int = Field(
        default=32, ge=1, le=128, description="Batch size for embeddings"
    )

    # ========================================
    # Vector Database (ChromaDB)
    # ========================================
    chroma_persist_directory: Path = Field(
        default=PROJECT_ROOT / "data" / "chroma_db",
        description="ChromaDB persistence directory",
    )
    chroma_collection_name: str = Field(
        default="documents_collection", description="ChromaDB collection name"
    )

    # ========================================
    # RAG Pipeline Settings
    # ========================================
    chunk_size: int = Field(
        default=1000, ge=100, le=2000, description="Text chunk size in characters"
    )
    chunk_overlap: int = Field(
        default=200, ge=0, le=500, description="Overlap between chunks"
    )
    retrieval_top_k: int = Field(
        default=5, ge=1, le=20, description="Number of chunks to retrieve"
    )
    similarity_threshold: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Minimum similarity score"
    )

    # ========================================
    # Application Settings
    # ========================================
    app_name: str = Field(default="MultiSource RAG System", description="App name")
    app_version: str = Field(default="1.0.0", description="App version")
    debug: bool = Field(default=False, description="Debug mode")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", description="Logging level"
    )

    # ========================================
    # API Settings (FastAPI)
    # ========================================
    api_host: str = Field(default="0.0.0.0", description="API host")
    api_port: int = Field(default=8000, ge=1024, le=65535, description="API port")
    api_workers: int = Field(default=1, ge=1, le=8, description="Number of workers")

    # ========================================
    # Frontend Settings (Streamlit)
    # ========================================
    frontend_port: int = Field(
        default=8501, ge=1024, le=65535, description="Streamlit port"
    )

    # ========================================
    # File Upload Settings
    # ========================================
    max_upload_size_mb: int = Field(
        default=10, ge=1, le=100, description="Max upload size in MB"
    )
    upload_directory: Path = Field(
        default=PROJECT_ROOT / "data" / "uploaded_docs",
        description="Upload directory",
    )

    # ========================================
    # Monitoring & Logging
    # ========================================
    log_file: Path = Field(
        default=PROJECT_ROOT / "logs" / "app.log", description="Log file path"
    )
    log_rotation: str = Field(default="10 MB", description="Log rotation size")
    log_retention: str = Field(default="30 days", description="Log retention period")

    @field_validator("chroma_persist_directory", "upload_directory", "log_file")
    @classmethod
    def create_directories(cls, v: Path) -> Path:
        """Ensure directories exist."""
        if v.suffix:  # If it's a file path, create parent directory
            v.parent.mkdir(parents=True, exist_ok=True)
        else:  # If it's a directory path
            v.mkdir(parents=True, exist_ok=True)
        return v

    @field_validator("chunk_overlap")
    @classmethod
    def validate_overlap(cls, v: int, info) -> int:
        """Ensure chunk overlap is less than chunk size."""
        chunk_size = info.data.get("chunk_size", 1000)
        if v >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        return v

    def get_api_key(self) -> str:
        """Get the API key for the selected LLM provider."""
        provider_keys = {
            "mistral": self.mistral_api_key,
            "openai": self.openai_api_key,
            "anthropic": self.anthropic_api_key,
        }
        key = provider_keys.get(self.llm_provider)
        if not key:
            raise ValueError(
                f"API key not configured for provider: {self.llm_provider}"
            )
        return key

    @property
    def max_upload_size_bytes(self) -> int:
        """Convert max upload size to bytes."""
        return self.max_upload_size_mb * 1024 * 1024


# Global settings instance
settings = Settings()


if __name__ == "__main__":
    """Test configuration loading."""
    print("=== Configuration Loaded ===")
    print(f"App: {settings.app_name} v{settings.app_version}")
    print(f"LLM Provider: {settings.llm_provider}")
    print(f"LLM Model: {settings.llm_model}")
    print(f"Embedding Model: {settings.embedding_model}")
    print(f"Embedding Device: {settings.embedding_device}")
    print(f"ChromaDB Directory: {settings.chroma_persist_directory}")
    print(f"Chunk Size: {settings.chunk_size}")
    print(f"Top K Retrieval: {settings.retrieval_top_k}")
    print(f"Debug Mode: {settings.debug}")
    print("\n✅ Configuration is valid!")
