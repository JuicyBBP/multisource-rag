"""
RAG (Retrieval-Augmented Generation) service for MultiSource RAG System.
Orchestrates document loading, chunking, embedding, storage, and LLM generation.
"""

from typing import List, Dict, Any
from pathlib import Path
from loguru import logger

from src.config import settings
from src.services.document_loader import DocumentLoader, Document
from src.services.text_chunker import TextChunker, Chunk
from src.services.embeddings_service import EmbeddingsService
from src.services.vector_store import VectorStore


class RAGService:
    """Main RAG service that coordinates all components."""

    def __init__(
        self,
        collection_name: str | None = None,
        reset_vector_store: bool = False,
    ):
        """Initialize RAG service.

        Args:
            collection_name: Name for ChromaDB collection (default from settings)
            reset_vector_store: If True, reset the vector store on initialization
        """
        logger.info("Initializing RAG Service...")

        # Initialize components
        self.document_loader = DocumentLoader()
        self.text_chunker = TextChunker()
        self.embeddings_service = EmbeddingsService()
        self.vector_store = VectorStore(
            collection_name=collection_name,
            reset=reset_vector_store,
        )

        # Initialize LLM client based on provider
        self._init_llm_client()

        logger.info(
            f"RAG Service initialized: provider={settings.llm_provider}, "
            f"model={settings.llm_model}, collection={self.vector_store.collection_name}"
        )

    def _init_llm_client(self):
        """Initialize LLM client based on configured provider."""
        try:
            if settings.llm_provider == "mistral":
                from mistralai import Mistral
                self.llm_client = Mistral(api_key=settings.mistral_api_key)
                logger.info("Initialized Mistral LLM client")

            elif settings.llm_provider == "openai":
                from openai import OpenAI
                self.llm_client = OpenAI(api_key=settings.openai_api_key)
                logger.info("Initialized OpenAI LLM client")

            elif settings.llm_provider == "anthropic":
                from anthropic import Anthropic
                self.llm_client = Anthropic(api_key=settings.anthropic_api_key)
                logger.info("Initialized Anthropic LLM client")

            else:
                raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")

        except Exception as e:
            logger.error(f"Error initializing LLM client: {e}")
            raise

    def ingest_document(
        self,
        file_path: Path | str,
        show_progress: bool = True,
    ) -> Dict[str, Any]:
        """Ingest a document into the RAG system.

        This loads, chunks, embeds, and stores the document.

        Args:
            file_path: Path to document file
            show_progress: Whether to show progress during processing

        Returns:
            Dictionary with ingestion statistics
        """
        file_path = Path(file_path)
        logger.info(f"Ingesting document: {file_path}")

        try:
            # 1. Load document
            document = self.document_loader.load(file_path)
            logger.info(f"Loaded document: {len(document.content)} characters")

            # 2. Chunk document
            chunks = self.text_chunker.chunk_document(document)
            logger.info(f"Created {len(chunks)} chunks")

            # 3. Generate embeddings
            embeddings = self.embeddings_service.encode_chunks(
                chunks,
                show_progress=show_progress,
            )
            logger.info(f"Generated {len(embeddings)} embeddings")

            # 4. Store in vector database
            self.vector_store.add_chunks(chunks, embeddings)
            logger.info(f"Stored chunks in vector database")

            # Statistics
            stats = {
                "file_name": file_path.name,
                "file_path": str(file_path),
                "num_characters": len(document.content),
                "num_chunks": len(chunks),
                "avg_chunk_size": sum(len(c.content) for c in chunks) // len(chunks),
                "status": "success",
            }

            logger.info(f"Document ingested successfully: {stats}")
            return stats

        except Exception as e:
            logger.error(f"Error ingesting document {file_path}: {e}")
            raise

    def ingest_documents(
        self,
        file_paths: List[Path | str],
        show_progress: bool = True,
    ) -> List[Dict[str, Any]]:
        """Ingest multiple documents.

        Args:
            file_paths: List of document file paths
            show_progress: Whether to show progress

        Returns:
            List of ingestion statistics for each document
        """
        results = []

        for file_path in file_paths:
            try:
                stats = self.ingest_document(file_path, show_progress=show_progress)
                results.append(stats)
            except Exception as e:
                logger.error(f"Failed to ingest {file_path}: {e}")
                results.append({
                    "file_path": str(file_path),
                    "status": "error",
                    "error": str(e),
                })

        logger.info(
            f"Ingested {len([r for r in results if r['status'] == 'success'])}/{len(file_paths)} documents"
        )

        return results

    def query(
        self,
        question: str,
        n_results: int | None = None,
        min_similarity: float | None = None,
        include_sources: bool = True,
    ) -> Dict[str, Any]:
        """Query the RAG system with a question.

        Args:
            question: Question to ask
            n_results: Number of chunks to retrieve (default from settings)
            min_similarity: Minimum similarity threshold (default from settings)
            include_sources: Whether to include source chunks in response

        Returns:
            Dictionary with answer and metadata
        """
        logger.info(f"Processing query: '{question}'")

        try:
            # 1. Retrieve relevant chunks
            search_results = self.vector_store.search_by_text(
                query_text=question,
                embeddings_service=self.embeddings_service,
                n_results=n_results,
                min_similarity=min_similarity,
            )

            if not search_results:
                logger.warning("No relevant chunks found for query")
                return {
                    "question": question,
                    "answer": "Je n'ai pas trouvé d'informations pertinentes dans les documents pour répondre à cette question.",
                    "sources": [],
                    "num_sources": 0,
                }

            logger.info(f"Retrieved {len(search_results)} relevant chunks")

            # 2. Build context from retrieved chunks
            context = self._build_context(search_results)

            # 3. Generate answer using LLM
            answer = self._generate_answer(question, context)

            # 4. Prepare response
            response = {
                "question": question,
                "answer": answer,
                "num_sources": len(search_results),
            }

            if include_sources:
                response["sources"] = [
                    {
                        "document": result["document"],
                        "similarity": result["similarity"],
                        "source": result["metadata"].get("original_metadata.file_name", "unknown"),
                    }
                    for result in search_results
                ]

            logger.info("Query processed successfully")
            return response

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            raise

    def _build_context(self, search_results: List[Dict[str, Any]]) -> str:
        """Build context string from search results.

        Args:
            search_results: List of search results from vector store

        Returns:
            Formatted context string
        """
        context_parts = []

        for idx, result in enumerate(search_results, 1):
            source = result["metadata"].get("original_metadata.file_name", "unknown")
            similarity = result["similarity"]
            document = result["document"]

            context_parts.append(
                f"[Source {idx} - {source} (similarité: {similarity:.2f})]\n{document}"
            )

        context = "\n\n".join(context_parts)
        return context

    def _generate_answer(self, question: str, context: str) -> str:
        """Generate answer using LLM.

        Args:
            question: User question
            context: Retrieved context

        Returns:
            Generated answer
        """
        # Build prompt
        system_prompt = """Tu es un assistant intelligent qui répond aux questions en te basant uniquement sur le contexte fourni.

Instructions:
1. Réponds de manière concise et précise à la question
2. Base ta réponse UNIQUEMENT sur les informations du contexte
3. Si le contexte ne contient pas assez d'informations, dis-le clairement
4. Cite les sources pertinentes quand c'est possible
5. Réponds en français"""

        user_prompt = f"""Contexte:
{context}

Question: {question}

Réponds à la question en te basant sur le contexte ci-dessus."""

        try:
            if settings.llm_provider == "mistral":
                response = self.llm_client.chat.complete(
                    model=settings.llm_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=settings.llm_temperature,
                    max_tokens=settings.llm_max_tokens,
                )
                answer = response.choices[0].message.content

            elif settings.llm_provider == "openai":
                response = self.llm_client.chat.completions.create(
                    model=settings.llm_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=settings.llm_temperature,
                    max_tokens=settings.llm_max_tokens,
                )
                answer = response.choices[0].message.content

            elif settings.llm_provider == "anthropic":
                response = self.llm_client.messages.create(
                    model=settings.llm_model,
                    system=system_prompt,
                    messages=[
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=settings.llm_temperature,
                    max_tokens=settings.llm_max_tokens,
                )
                answer = response.content[0].text

            else:
                raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")

            logger.info("Generated answer using LLM")
            return answer.strip()

        except Exception as e:
            logger.error(f"Error generating answer with LLM: {e}")
            raise

    def get_stats(self) -> Dict[str, Any]:
        """Get RAG system statistics.

        Returns:
            Dictionary with system statistics
        """
        vector_stats = self.vector_store.get_collection_stats()
        embedding_info = self.embeddings_service.get_model_info()

        stats = {
            "vector_store": vector_stats,
            "embeddings": embedding_info,
            "llm_provider": settings.llm_provider,
            "llm_model": settings.llm_model,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "retrieval_top_k": settings.retrieval_top_k,
        }

        return stats

    def delete_document(self, source: str) -> None:
        """Delete all chunks from a specific document.

        Args:
            source: Source identifier (filename)
        """
        self.vector_store.delete_by_source(source)
        logger.info(f"Deleted document: {source}")

    def reset(self) -> None:
        """Reset the RAG system (clear all documents)."""
        self.vector_store.reset_collection()
        logger.info("RAG system reset")


if __name__ == "__main__":
    """Test RAG service."""
    import sys

    # Configure logger for testing
    logger.remove()
    logger.add(sys.stdout, level="INFO")

    print("=== RAG Service Test ===\n")

    # Create test documents
    test_dir = Path("data/test_docs")
    test_dir.mkdir(parents=True, exist_ok=True)

    test_file = test_dir / "rag_test.txt"
    test_file.write_text(
        """Machine Learning Fundamentals

Machine learning is a subset of artificial intelligence that enables computers to learn from data without being explicitly programmed. It uses algorithms that iteratively learn from data to improve their performance.

There are three main types of machine learning:
1. Supervised Learning: Uses labeled data to train models
2. Unsupervised Learning: Finds patterns in unlabeled data
3. Reinforcement Learning: Learns through trial and error

Deep learning is a specialized form of machine learning that uses neural networks with multiple layers. These networks can learn complex patterns and representations from large amounts of data."""
    )

    # Initialize RAG service (reset for testing)
    print("Initializing RAG Service...\n")
    rag = RAGService(
        collection_name="rag_test_collection",
        reset_vector_store=True,
    )

    # Ingest document
    print("Ingesting test document...\n")
    ingest_stats = rag.ingest_document(test_file, show_progress=False)
    print(f"Ingestion Stats:")
    for key, value in ingest_stats.items():
        print(f"  {key}: {value}")

    # Get system stats
    print("\n\nSystem Stats:")
    stats = rag.get_stats()
    print(f"  Total chunks: {stats['vector_store']['total_chunks']}")
    print(f"  Embedding model: {stats['embeddings']['model_name']}")
    print(f"  LLM: {stats['llm_provider']} - {stats['llm_model']}")

    # Test query (note: this will fail without valid API key, but structure is correct)
    print("\n\nTesting query structure (without actual LLM call)...")
    try:
        question = "What are the three types of machine learning?"
        print(f"Question: {question}")

        # This will fail if no API key, but shows the structure works
        response = rag.query(question, include_sources=True)

        print(f"\nAnswer: {response['answer']}")
        print(f"\nSources used: {response['num_sources']}")

        if response.get("sources"):
            print("\nSource details:")
            for idx, source in enumerate(response["sources"], 1):
                print(f"  {idx}. Similarity: {source['similarity']:.4f}")
                print(f"     Text: {source['document'][:80]}...")

    except Exception as e:
        print(f"Query test failed (expected if no API key): {type(e).__name__}")
        print("This is normal for testing without valid credentials")

    print("\n✅ RAG service test completed!")
