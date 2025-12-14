"""
Text chunking module for MultiSource RAG System.
Splits documents into smaller chunks for efficient embedding and retrieval.
"""

from typing import List, Dict, Any
from loguru import logger

from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import settings
from src.services.document_loader import Document


class Chunk:
    """Represents a text chunk with metadata."""

    def __init__(
        self,
        content: str,
        chunk_id: str,
        source: str,
        metadata: Dict[str, Any] | None = None,
    ):
        self.content = content
        self.chunk_id = chunk_id
        self.source = source
        self.metadata = metadata or {}

    def __repr__(self) -> str:
        preview = self.content[:50].replace("\n", " ")
        return (
            f"Chunk(id='{self.chunk_id}', source='{self.source}', "
            f"content='{preview}...', metadata={self.metadata})"
        )


class TextChunker:
    """Splits text documents into chunks for embedding."""

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        separators: List[str] | None = None,
    ):
        """Initialize text chunker.

        Args:
            chunk_size: Maximum chunk size in characters (default from settings)
            chunk_overlap: Overlap between chunks in characters (default from settings)
            separators: List of separators to use for splitting (default: hierarchical)
        """
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap

        # Hierarchical separators: try to split on larger units first
        self.separators = separators or [
            "\n\n\n",  # Multiple newlines (section breaks)
            "\n\n",  # Paragraph breaks
            "\n",  # Single newline
            ". ",  # Sentence endings
            "! ",  # Exclamation
            "? ",  # Question
            "; ",  # Semicolon
            ", ",  # Comma
            " ",  # Space
            "",  # Character-level fallback
        ]

        # Initialize LangChain splitter
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            length_function=len,
            is_separator_regex=False,
        )

        logger.info(
            f"TextChunker initialized: chunk_size={self.chunk_size}, "
            f"chunk_overlap={self.chunk_overlap}"
        )

    def chunk_text(self, text: str, source: str = "unknown") -> List[Chunk]:
        """Split text into chunks.

        Args:
            text: Text to split
            source: Source identifier for the text

        Returns:
            List of Chunk objects
        """
        if not text or not text.strip():
            logger.warning(f"Empty text provided for chunking from source: {source}")
            return []

        try:
            # Split text using LangChain splitter
            text_chunks = self.splitter.split_text(text)

            # Create Chunk objects with metadata
            chunks = []
            for idx, chunk_text in enumerate(text_chunks):
                chunk_id = f"{source}_chunk_{idx}"

                chunk = Chunk(
                    content=chunk_text,
                    chunk_id=chunk_id,
                    source=source,
                    metadata={
                        "chunk_index": idx,
                        "total_chunks": len(text_chunks),
                        "chunk_size": len(chunk_text),
                        "char_start": sum(len(c) for c in text_chunks[:idx]),
                        "char_end": sum(len(c) for c in text_chunks[: idx + 1]),
                    },
                )
                chunks.append(chunk)

            logger.info(
                f"Split text from '{source}' into {len(chunks)} chunks "
                f"(avg size: {sum(len(c.content) for c in chunks) // len(chunks)} chars)"
            )

            return chunks

        except Exception as e:
            logger.error(f"Error chunking text from {source}: {e}")
            raise

    def chunk_document(self, document: Document) -> List[Chunk]:
        """Split a Document into chunks, preserving metadata.

        Args:
            document: Document object to chunk

        Returns:
            List of Chunk objects
        """
        try:
            # Use document source as identifier
            source_id = document.metadata.get("file_name", document.source)

            # Split the document content
            chunks = self.chunk_text(document.content, source=source_id)

            # Enrich chunk metadata with document metadata
            for chunk in chunks:
                chunk.metadata.update(
                    {
                        "original_source": document.source,
                        "original_metadata": document.metadata,
                    }
                )

            logger.info(
                f"Chunked document '{source_id}': {len(chunks)} chunks created"
            )

            return chunks

        except Exception as e:
            logger.error(f"Error chunking document {document.source}: {e}")
            raise

    def chunk_multiple_documents(
        self, documents: List[Document]
    ) -> List[Chunk]:
        """Split multiple documents into chunks.

        Args:
            documents: List of Document objects

        Returns:
            List of all Chunk objects from all documents
        """
        all_chunks = []

        for doc in documents:
            try:
                chunks = self.chunk_document(doc)
                all_chunks.extend(chunks)

            except Exception as e:
                logger.error(f"Failed to chunk document {doc.source}: {e}")
                # Continue with other documents
                continue

        logger.info(
            f"Chunked {len(documents)} documents into {len(all_chunks)} total chunks"
        )

        return all_chunks

    def get_chunk_stats(self, chunks: List[Chunk]) -> Dict[str, Any]:
        """Calculate statistics about chunks.

        Args:
            chunks: List of chunks

        Returns:
            Dictionary with chunk statistics
        """
        if not chunks:
            return {
                "num_chunks": 0,
                "total_chars": 0,
                "avg_chunk_size": 0,
                "min_chunk_size": 0,
                "max_chunk_size": 0,
            }

        chunk_sizes = [len(chunk.content) for chunk in chunks]

        return {
            "num_chunks": len(chunks),
            "total_chars": sum(chunk_sizes),
            "avg_chunk_size": sum(chunk_sizes) // len(chunks),
            "min_chunk_size": min(chunk_sizes),
            "max_chunk_size": max(chunk_sizes),
            "unique_sources": len(set(chunk.source for chunk in chunks)),
        }


if __name__ == "__main__":
    """Test text chunker."""
    import sys

    # Configure logger for testing
    logger.remove()
    logger.add(sys.stdout, level="INFO")

    print("=== Text Chunker Test ===\n")

    # Create test document
    from pathlib import Path
    from src.services.document_loader import DocumentLoader

    # Create test directory and file
    test_dir = Path("data/test_docs")
    test_dir.mkdir(parents=True, exist_ok=True)

    test_file = test_dir / "chunking_test.txt"
    test_file.write_text(
        """This is the first paragraph. It contains some text that will be chunked.

This is the second paragraph. It has more content to demonstrate chunking.

This is the third paragraph with even more text. The chunker should split this appropriately.

Fourth paragraph here. We want to test how the chunker handles multiple paragraphs and maintains context.

Fifth and final paragraph. This should show how overlaps work between chunks to preserve context across boundaries."""
    )

    # Load document
    doc = DocumentLoader.load(test_file)
    print(f"Loaded document: {doc.metadata['file_name']}")
    print(f"Content length: {len(doc.content)} characters\n")

    # Initialize chunker with small chunk size for testing
    chunker = TextChunker(chunk_size=150, chunk_overlap=30)

    # Chunk the document
    chunks = chunker.chunk_document(doc)

    print(f"\nCreated {len(chunks)} chunks:\n")
    for chunk in chunks:
        print(f"Chunk {chunk.metadata['chunk_index']}:")
        print(f"  Size: {len(chunk.content)} chars")
        print(f"  Content: {chunk.content[:80]}...")
        print()

    # Get statistics
    stats = chunker.get_chunk_stats(chunks)
    print(f"Chunk Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\n✅ Text chunker test completed!")
