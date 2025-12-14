"""
Embeddings service for MultiSource RAG System.
Generates vector embeddings from text using Sentence Transformers.
"""

from typing import List, Dict, Any
import numpy as np
import torch
from loguru import logger
from sentence_transformers import SentenceTransformer

from src.config import settings
from src.services.text_chunker import Chunk


class EmbeddingsService:
    """Service for generating text embeddings."""

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        batch_size: int | None = None,
    ):
        """Initialize embeddings service.

        Args:
            model_name: Name of the sentence transformer model (default from settings)
            device: Device to use ('cuda' or 'cpu', default from settings)
            batch_size: Batch size for encoding (default from settings)
        """
        self.model_name = model_name or settings.embedding_model
        self.batch_size = batch_size or settings.embedding_batch_size

        # Determine device
        if device:
            self.device = device
        else:
            # Use settings device if CUDA is available, else fallback to CPU
            if settings.embedding_device == "cuda" and torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
                if settings.embedding_device == "cuda":
                    logger.warning(
                        "CUDA requested but not available, falling back to CPU"
                    )

        # Load model
        logger.info(f"Loading embedding model: {self.model_name} on {self.device}")
        self.model = SentenceTransformer(self.model_name, device=self.device)

        # Get embedding dimension
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

        logger.info(
            f"EmbeddingsService initialized: model={self.model_name}, "
            f"device={self.device}, dim={self.embedding_dim}, "
            f"batch_size={self.batch_size}"
        )

    def encode_text(
        self,
        text: str,
        normalize: bool = True,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Generate embedding for a single text.

        Args:
            text: Text to encode
            normalize: Whether to normalize embeddings to unit length
            show_progress: Whether to show progress bar

        Returns:
            Numpy array of embedding vector
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for encoding")
            return np.zeros(self.embedding_dim)

        try:
            embedding = self.model.encode(
                text,
                normalize_embeddings=normalize,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
            )
            return embedding

        except Exception as e:
            logger.error(f"Error encoding text: {e}")
            raise

    def encode_texts(
        self,
        texts: List[str],
        normalize: bool = True,
        show_progress: bool = True,
    ) -> List[np.ndarray]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to encode
            normalize: Whether to normalize embeddings to unit length
            show_progress: Whether to show progress bar

        Returns:
            List of numpy arrays (embeddings)
        """
        if not texts:
            logger.warning("Empty text list provided for encoding")
            return []

        # Filter out empty texts
        valid_texts = [t for t in texts if t and t.strip()]
        if len(valid_texts) < len(texts):
            logger.warning(
                f"Filtered out {len(texts) - len(valid_texts)} empty texts"
            )

        if not valid_texts:
            return [np.zeros(self.embedding_dim) for _ in texts]

        try:
            logger.info(f"Encoding {len(valid_texts)} texts (batch_size={self.batch_size})")

            embeddings = self.model.encode(
                valid_texts,
                batch_size=self.batch_size,
                normalize_embeddings=normalize,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
            )

            # Convert to list of arrays
            embeddings_list = [emb for emb in embeddings]

            logger.info(
                f"Successfully encoded {len(embeddings_list)} texts "
                f"(dim={self.embedding_dim})"
            )

            return embeddings_list

        except Exception as e:
            logger.error(f"Error encoding texts: {e}")
            raise

    def encode_chunk(
        self,
        chunk: Chunk,
        normalize: bool = True,
    ) -> np.ndarray:
        """Generate embedding for a Chunk object.

        Args:
            chunk: Chunk object to encode
            normalize: Whether to normalize embedding

        Returns:
            Numpy array of embedding vector
        """
        try:
            embedding = self.encode_text(
                chunk.content,
                normalize=normalize,
                show_progress=False,
            )

            # Add embedding info to chunk metadata
            chunk.metadata["embedding_model"] = self.model_name
            chunk.metadata["embedding_dim"] = self.embedding_dim
            chunk.metadata["has_embedding"] = True

            return embedding

        except Exception as e:
            logger.error(f"Error encoding chunk {chunk.chunk_id}: {e}")
            raise

    def encode_chunks(
        self,
        chunks: List[Chunk],
        normalize: bool = True,
        show_progress: bool = True,
    ) -> List[np.ndarray]:
        """Generate embeddings for multiple Chunk objects.

        Args:
            chunks: List of Chunk objects to encode
            normalize: Whether to normalize embeddings
            show_progress: Whether to show progress bar

        Returns:
            List of numpy arrays (embeddings)
        """
        if not chunks:
            logger.warning("Empty chunk list provided for encoding")
            return []

        try:
            # Extract texts from chunks
            texts = [chunk.content for chunk in chunks]

            # Generate embeddings
            embeddings = self.encode_texts(
                texts,
                normalize=normalize,
                show_progress=show_progress,
            )

            # Update chunk metadata
            for chunk in chunks:
                chunk.metadata["embedding_model"] = self.model_name
                chunk.metadata["embedding_dim"] = self.embedding_dim
                chunk.metadata["has_embedding"] = True

            logger.info(f"Successfully encoded {len(chunks)} chunks")

            return embeddings

        except Exception as e:
            logger.error(f"Error encoding chunks: {e}")
            raise

    def get_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
    ) -> float:
        """Calculate cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Cosine similarity score (0 to 1)
        """
        try:
            # Ensure arrays are 1D
            emb1 = embedding1.flatten()
            emb2 = embedding2.flatten()

            # Calculate cosine similarity
            dot_product = np.dot(emb1, emb2)
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            similarity = dot_product / (norm1 * norm2)

            # Clamp to [0, 1] range (sometimes numerical errors cause values slightly outside)
            return float(np.clip(similarity, 0.0, 1.0))

        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            raise

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model.

        Returns:
            Dictionary with model information
        """
        return {
            "model_name": self.model_name,
            "embedding_dim": self.embedding_dim,
            "device": self.device,
            "batch_size": self.batch_size,
            "max_seq_length": self.model.max_seq_length,
        }


if __name__ == "__main__":
    """Test embeddings service."""
    import sys

    # Configure logger for testing
    logger.remove()
    logger.add(sys.stdout, level="INFO")

    print("=== Embeddings Service Test ===\n")

    # Initialize service
    embeddings_service = EmbeddingsService()

    # Print model info
    model_info = embeddings_service.get_model_info()
    print("Model Information:")
    for key, value in model_info.items():
        print(f"  {key}: {value}")
    print()

    # Test single text encoding
    test_text = "This is a test sentence for embedding generation."
    print(f"Test text: '{test_text}'")

    embedding = embeddings_service.encode_text(test_text)
    print(f"Generated embedding: shape={embedding.shape}, dtype={embedding.dtype}")
    print(f"First 10 values: {embedding[:10]}")
    print()

    # Test multiple texts encoding
    test_texts = [
        "Machine learning is fascinating.",
        "Natural language processing uses neural networks.",
        "Vector embeddings capture semantic meaning.",
    ]
    print(f"Encoding {len(test_texts)} texts...")

    embeddings = embeddings_service.encode_texts(test_texts, show_progress=False)
    print(f"Generated {len(embeddings)} embeddings")
    print(f"Embedding shapes: {[emb.shape for emb in embeddings]}")
    print()

    # Test similarity
    similarity_1_2 = embeddings_service.get_similarity(embeddings[0], embeddings[1])
    similarity_1_3 = embeddings_service.get_similarity(embeddings[0], embeddings[2])
    similarity_2_3 = embeddings_service.get_similarity(embeddings[1], embeddings[2])

    print("Similarity scores:")
    print(f"  Text 1 vs Text 2: {similarity_1_2:.4f}")
    print(f"  Text 1 vs Text 3: {similarity_1_3:.4f}")
    print(f"  Text 2 vs Text 3: {similarity_2_3:.4f}")
    print()

    # Test with chunks
    from pathlib import Path
    from src.services.document_loader import DocumentLoader
    from src.services.text_chunker import TextChunker

    # Create and load test document
    test_dir = Path("data/test_docs")
    test_file = test_dir / "embedding_test.txt"
    test_file.write_text(
        "Artificial intelligence is transforming technology. "
        "Machine learning algorithms learn from data. "
        "Deep learning uses neural networks with multiple layers."
    )

    doc = DocumentLoader.load(test_file)
    chunker = TextChunker(chunk_size=50, chunk_overlap=10)
    chunks = chunker.chunk_document(doc)

    print(f"Created {len(chunks)} chunks from test document")

    # Encode chunks
    chunk_embeddings = embeddings_service.encode_chunks(chunks, show_progress=False)
    print(f"Generated {len(chunk_embeddings)} chunk embeddings")
    print(f"Chunk metadata updated: {chunks[0].metadata.get('has_embedding')}")

    print("\n✅ Embeddings service test completed!")
