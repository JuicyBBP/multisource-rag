"""
Vector store service for MultiSource RAG System.
Manages document storage and retrieval using ChromaDB.
"""

from typing import List, Dict, Any, Tuple
from pathlib import Path
import numpy as np
from loguru import logger

import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.api.models.Collection import Collection

from src.config import settings
from src.services.text_chunker import Chunk


class VectorStore:
    """Vector database service using ChromaDB."""

    def __init__(
        self,
        collection_name: str | None = None,
        persist_directory: Path | None = None,
        reset: bool = False,
    ):
        """Initialize vector store.

        Args:
            collection_name: Name of the collection (default from settings)
            persist_directory: Directory for persistence (default from settings)
            reset: If True, delete existing collection and start fresh
        """
        self.collection_name = collection_name or settings.chroma_collection_name
        self.persist_directory = persist_directory or settings.chroma_persist_directory

        # Ensure persist directory exists
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB client with persistence
        logger.info(f"Initializing ChromaDB at: {self.persist_directory}")
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True,
            ),
        )

        # Get or create collection
        if reset:
            logger.warning(f"Resetting collection: {self.collection_name}")
            try:
                self.client.delete_collection(name=self.collection_name)
            except Exception:
                pass  # Collection might not exist

        self.collection: Collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "MultiSource RAG document collection"},
        )

        logger.info(
            f"VectorStore initialized: collection='{self.collection_name}', "
            f"count={self.collection.count()}"
        )

    @staticmethod
    def _flatten_metadata(metadata: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
        """Flatten nested metadata dictionaries for ChromaDB.

        ChromaDB only accepts str, int, float, bool, or None as metadata values.
        This function flattens nested dicts into dot-notation keys.

        Args:
            metadata: Metadata dictionary to flatten
            prefix: Prefix for nested keys

        Returns:
            Flattened metadata dictionary
        """
        flattened = {}

        for key, value in metadata.items():
            new_key = f"{prefix}.{key}" if prefix else key

            if isinstance(value, dict):
                # Recursively flatten nested dicts
                flattened.update(VectorStore._flatten_metadata(value, new_key))
            elif isinstance(value, (str, int, float, bool)) or value is None:
                # Keep valid types as-is
                flattened[new_key] = value
            else:
                # Convert other types to string
                flattened[new_key] = str(value)

        return flattened

    def add_chunks(
        self,
        chunks: List[Chunk],
        embeddings: List[np.ndarray],
    ) -> None:
        """Add chunks with embeddings to the vector store.

        Args:
            chunks: List of Chunk objects
            embeddings: List of embedding vectors (same length as chunks)

        Raises:
            ValueError: If chunks and embeddings have different lengths
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Number of chunks ({len(chunks)}) must match "
                f"number of embeddings ({len(embeddings)})"
            )

        if not chunks:
            logger.warning("Empty chunks list provided")
            return

        try:
            # Prepare data for ChromaDB
            ids = [chunk.chunk_id for chunk in chunks]
            documents = [chunk.content for chunk in chunks]

            # Flatten metadata to ensure ChromaDB compatibility
            metadatas = [self._flatten_metadata(chunk.metadata) for chunk in chunks]

            embeddings_list = [emb.tolist() for emb in embeddings]

            # Add to collection
            self.collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings_list,
                metadatas=metadatas,
            )

            logger.info(
                f"Added {len(chunks)} chunks to collection '{self.collection_name}' "
                f"(total: {self.collection.count()})"
            )

        except Exception as e:
            logger.error(f"Error adding chunks to vector store: {e}")
            raise

    def search(
        self,
        query_embedding: np.ndarray,
        n_results: int | None = None,
        min_similarity: float | None = None,
        filter_metadata: Dict[str, Any] | None = None,
    ) -> List[Dict[str, Any]]:
        """Search for similar chunks using query embedding.

        Args:
            query_embedding: Query embedding vector
            n_results: Number of results to return (default from settings)
            min_similarity: Minimum similarity threshold (default from settings)
            filter_metadata: Metadata filter dict (e.g., {"source": "doc1.pdf"})

        Returns:
            List of search results with documents, metadata, and distances
        """
        n_results = n_results or settings.retrieval_top_k
        min_similarity = min_similarity or settings.similarity_threshold

        try:
            # Query ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=n_results,
                where=filter_metadata,
                include=["documents", "metadatas", "distances"],
            )

            # Process results
            search_results = []
            for idx in range(len(results["ids"][0])):
                distance = results["distances"][0][idx]

                # Convert distance to similarity (ChromaDB uses L2 distance)
                # For normalized embeddings: similarity = 1 - (distance^2 / 2)
                similarity = 1 - (distance**2 / 2)

                # Filter by similarity threshold
                if similarity < min_similarity:
                    continue

                result = {
                    "id": results["ids"][0][idx],
                    "document": results["documents"][0][idx],
                    "metadata": results["metadatas"][0][idx],
                    "distance": distance,
                    "similarity": similarity,
                }
                search_results.append(result)

            logger.info(
                f"Search returned {len(search_results)} results "
                f"(threshold: {min_similarity:.2f})"
            )

            return search_results

        except Exception as e:
            logger.error(f"Error searching vector store: {e}")
            raise

    def search_by_text(
        self,
        query_text: str,
        embeddings_service,
        n_results: int | None = None,
        min_similarity: float | None = None,
        filter_metadata: Dict[str, Any] | None = None,
    ) -> List[Dict[str, Any]]:
        """Search using query text (automatically generates embedding).

        Args:
            query_text: Query text
            embeddings_service: EmbeddingsService instance
            n_results: Number of results to return
            min_similarity: Minimum similarity threshold
            filter_metadata: Metadata filter dict

        Returns:
            List of search results
        """
        # Generate query embedding
        query_embedding = embeddings_service.encode_text(query_text)

        # Search using embedding
        return self.search(
            query_embedding=query_embedding,
            n_results=n_results,
            min_similarity=min_similarity,
            filter_metadata=filter_metadata,
        )

    def get_by_ids(self, ids: List[str]) -> Dict[str, Any]:
        """Retrieve chunks by their IDs.

        Args:
            ids: List of chunk IDs

        Returns:
            Dictionary with documents and metadata
        """
        try:
            results = self.collection.get(
                ids=ids,
                include=["documents", "metadatas"],
            )

            logger.info(f"Retrieved {len(results['ids'])} chunks by ID")

            return results

        except Exception as e:
            logger.error(f"Error retrieving chunks by ID: {e}")
            raise

    def delete_by_ids(self, ids: List[str]) -> None:
        """Delete chunks by their IDs.

        Args:
            ids: List of chunk IDs to delete
        """
        try:
            self.collection.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} chunks from collection")

        except Exception as e:
            logger.error(f"Error deleting chunks: {e}")
            raise

    def delete_by_source(self, source: str) -> None:
        """Delete all chunks from a specific source.

        Args:
            source: Source identifier (e.g., filename)
        """
        try:
            # Get all chunks from this source
            results = self.collection.get(
                where={"original_metadata.file_name": source},
                include=["documents"],
            )

            if results["ids"]:
                self.collection.delete(ids=results["ids"])
                logger.info(f"Deleted {len(results['ids'])} chunks from source '{source}'")
            else:
                logger.info(f"No chunks found from source '{source}'")

        except Exception as e:
            logger.error(f"Error deleting chunks by source: {e}")
            raise

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection.

        Returns:
            Dictionary with collection statistics
        """
        try:
            count = self.collection.count()

            # Get sample of metadata to analyze sources
            sample = self.collection.get(
                limit=min(1000, count) if count > 0 else 1,
                include=["metadatas"],
            )

            unique_sources = set()
            if sample["metadatas"]:
                for metadata in sample["metadatas"]:
                    # Use flattened metadata key
                    source = metadata.get("original_metadata.file_name")
                    if source:
                        unique_sources.add(source)

            stats = {
                "collection_name": self.collection_name,
                "total_chunks": count,
                "unique_sources_sampled": len(unique_sources),
                "persist_directory": str(self.persist_directory),
            }

            logger.info(f"Collection stats: {stats}")

            return stats

        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            raise

    def reset_collection(self) -> None:
        """Delete all data in the collection."""
        try:
            logger.warning(f"Resetting collection: {self.collection_name}")
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "MultiSource RAG document collection"},
            )
            logger.info("Collection reset successfully")

        except Exception as e:
            logger.error(f"Error resetting collection: {e}")
            raise


if __name__ == "__main__":
    """Test vector store."""
    import sys

    # Configure logger for testing
    logger.remove()
    logger.add(sys.stdout, level="INFO")

    print("=== Vector Store Test ===\n")

    # Initialize services
    from src.services.document_loader import DocumentLoader
    from src.services.text_chunker import TextChunker
    from src.services.embeddings_service import EmbeddingsService

    # Create test documents
    from pathlib import Path

    test_dir = Path("data/test_docs")
    test_dir.mkdir(parents=True, exist_ok=True)

    test_file1 = test_dir / "vector_test1.txt"
    test_file1.write_text(
        "Machine learning is a subset of artificial intelligence. "
        "It focuses on teaching computers to learn from data."
    )

    test_file2 = test_dir / "vector_test2.txt"
    test_file2.write_text(
        "Natural language processing enables computers to understand human language. "
        "It uses neural networks and deep learning techniques."
    )

    # Load and process documents
    doc1 = DocumentLoader.load(test_file1)
    doc2 = DocumentLoader.load(test_file2)

    chunker = TextChunker(chunk_size=100, chunk_overlap=20)
    chunks = chunker.chunk_multiple_documents([doc1, doc2])

    print(f"Created {len(chunks)} chunks from {2} documents\n")

    # Generate embeddings
    embeddings_service = EmbeddingsService()
    embeddings = embeddings_service.encode_chunks(chunks, show_progress=False)

    print(f"Generated {len(embeddings)} embeddings\n")

    # Initialize vector store (with reset for testing)
    vector_store = VectorStore(
        collection_name="test_collection",
        reset=True,
    )

    # Add chunks to vector store
    vector_store.add_chunks(chunks, embeddings)

    # Get collection stats
    stats = vector_store.get_collection_stats()
    print(f"\nCollection Stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Test search with query
    query = "How do computers learn from data?"
    print(f"\nQuery: '{query}'")

    results = vector_store.search_by_text(
        query_text=query,
        embeddings_service=embeddings_service,
        n_results=3,
        min_similarity=0.0,  # Show all results for testing
    )

    print(f"\nSearch Results ({len(results)} found):")
    for idx, result in enumerate(results):
        print(f"\n{idx + 1}. Similarity: {result['similarity']:.4f}")
        print(f"   Document: {result['document'][:80]}...")
        print(f"   Source: {result['metadata'].get('original_metadata.file_name', 'unknown')}")

    print("\n✅ Vector store test completed!")
