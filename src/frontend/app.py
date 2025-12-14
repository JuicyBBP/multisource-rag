"""
Streamlit frontend for MultiSource RAG System.
"""

import streamlit as st
import requests
from pathlib import Path
import json

# Page configuration
st.set_page_config(
    page_title="MultiSource RAG System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# API configuration
API_BASE_URL = "http://localhost:8000/api/v1"


def check_api_health():
    """Check if API is running."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False


def get_system_stats():
    """Get system statistics from API."""
    try:
        response = requests.get(f"{API_BASE_URL}/stats")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching stats: {e}")
        return None


def upload_document(file):
    """Upload document to API."""
    try:
        files = {"file": (file.name, file.getvalue(), file.type)}
        response = requests.post(f"{API_BASE_URL}/ingest", files=files)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        error_detail = e.response.json() if e.response.content else str(e)
        raise Exception(f"Upload failed: {error_detail}")
    except Exception as e:
        raise Exception(f"Upload error: {str(e)}")


def query_rag(question, n_results=5, min_similarity=0.7, include_sources=True):
    """Query the RAG system."""
    try:
        payload = {
            "question": question,
            "n_results": n_results,
            "min_similarity": min_similarity,
            "include_sources": include_sources,
        }
        response = requests.post(f"{API_BASE_URL}/query", json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        error_detail = e.response.json() if e.response.content else str(e)
        raise Exception(f"Query failed: {error_detail}")
    except Exception as e:
        raise Exception(f"Query error: {str(e)}")


# Sidebar
with st.sidebar:
    st.title("🤖 MultiSource RAG")
    st.markdown("---")

    # API Status
    api_status = check_api_health()
    if api_status:
        st.success("✅ API Connected")
    else:
        st.error("❌ API Offline")
        st.warning("Make sure the FastAPI server is running on port 8000")
        st.code("python src/api/main.py", language="bash")
        st.stop()

    # Navigation
    st.markdown("### Navigation")
    page = st.radio(
        "Go to",
        ["🏠 Home", "📤 Upload Documents", "💬 Ask Questions", "📊 Statistics"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.caption("MultiSource RAG System v1.0.0")


# Main content
if page == "🏠 Home":
    st.title("Welcome to MultiSource RAG System")

    st.markdown("""
    ### 🎯 What is RAG?

    **Retrieval-Augmented Generation (RAG)** combines the power of:
    - 🔍 **Semantic Search**: Find relevant information in your documents
    - 🤖 **Large Language Models**: Generate accurate, contextual answers

    ### 🚀 How to Use

    1. **📤 Upload Documents**
        - Support for PDF, DOCX, TXT, and Markdown files
        - Documents are automatically processed and indexed

    2. **💬 Ask Questions**
        - Get AI-powered answers based on your documents
        - See source citations for transparency

    3. **📊 Monitor Statistics**
        - Track your document collection
        - View system performance metrics

    ### 🛠️ System Architecture
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info("**Document Processing**\n\n✓ Multi-format support\n\n✓ Intelligent chunking\n\n✓ Metadata preservation")

    with col2:
        st.info("**Vector Search**\n\n✓ GPU-accelerated embeddings\n\n✓ ChromaDB storage\n\n✓ Similarity search")

    with col3:
        st.info("**AI Generation**\n\n✓ Multiple LLM providers\n\n✓ Context-aware answers\n\n✓ Source citations")

    # Quick stats
    st.markdown("### 📈 Quick Stats")
    stats = get_system_stats()
    if stats:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Chunks", stats["vector_store"]["total_chunks"])
        with col2:
            st.metric("Documents", stats["vector_store"]["unique_sources_sampled"])
        with col3:
            st.metric("Embedding Dimension", stats["embeddings"]["embedding_dim"])
        with col4:
            st.metric("LLM Provider", stats["llm_provider"].upper())


elif page == "📤 Upload Documents":
    st.title("📤 Upload Documents")

    st.markdown("""
    Upload your documents to make them searchable with AI-powered question answering.

    **Supported formats:** PDF, DOCX, DOC, TXT, MD, Markdown
    """)

    # File uploader
    uploaded_files = st.file_uploader(
        "Choose files to upload",
        type=["pdf", "docx", "doc", "txt", "md", "markdown"],
        accept_multiple_files=True,
        help="Select one or more documents to upload",
    )

    if uploaded_files:
        st.markdown(f"### Selected Files ({len(uploaded_files)})")

        for file in uploaded_files:
            st.text(f"📄 {file.name} ({file.size / 1024:.1f} KB)")

        if st.button("🚀 Upload and Process", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()

            for idx, file in enumerate(uploaded_files):
                status_text.text(f"Processing {file.name}...")

                try:
                    result = upload_document(file)
                    st.success(f"✅ {file.name}: {result['stats']['num_chunks']} chunks created")

                except Exception as e:
                    st.error(f"❌ {file.name}: {str(e)}")

                progress_bar.progress((idx + 1) / len(uploaded_files))

            status_text.text("✨ All files processed!")
            st.balloons()


elif page == "💬 Ask Questions":
    st.title("💬 Ask Questions")

    st.markdown("""
    Ask questions about your uploaded documents and get AI-powered answers with source citations.
    """)

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Display sources if available
            if message.get("sources"):
                with st.expander(f"📚 Sources ({len(message['sources'])})"):
                    for idx, source in enumerate(message["sources"], 1):
                        st.markdown(f"**Source {idx}** - {source['source']} (similarity: {source['similarity']:.2%})")
                        st.text(source["document"][:200] + "..." if len(source["document"]) > 200 else source["document"])
                        st.markdown("---")

    # Query settings in sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Query Settings")
        n_results = st.slider("Number of sources", 1, 10, 5)
        min_similarity = st.slider("Minimum similarity", 0.0, 1.0, 0.7, 0.05)

    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = query_rag(
                        question=prompt,
                        n_results=n_results,
                        min_similarity=min_similarity,
                        include_sources=True,
                    )

                    st.markdown(response["answer"])

                    # Add assistant response to chat
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response["answer"],
                        "sources": response.get("sources", []),
                    })

                    # Display sources
                    if response.get("sources"):
                        with st.expander(f"📚 Sources ({len(response['sources'])})"):
                            for idx, source in enumerate(response["sources"], 1):
                                st.markdown(f"**Source {idx}** - {source['source']} (similarity: {source['similarity']:.2%})")
                                st.text(source["document"][:200] + "..." if len(source["document"]) > 200 else source["document"])
                                st.markdown("---")

                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                    })

    # Clear chat button
    if st.session_state.messages:
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()


elif page == "📊 Statistics":
    st.title("📊 System Statistics")

    # Get stats
    stats = get_system_stats()

    if stats:
        # Vector Store Stats
        st.markdown("### 🗄️ Vector Store")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Collection Name", stats["vector_store"]["collection_name"])
        with col2:
            st.metric("Total Chunks", stats["vector_store"]["total_chunks"])
        with col3:
            st.metric("Unique Documents", stats["vector_store"]["unique_sources_sampled"])

        st.text(f"📁 Storage: {stats['vector_store']['persist_directory']}")

        # Embeddings Model
        st.markdown("### 🧠 Embeddings Model")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Model", stats["embeddings"]["model_name"].split("/")[-1])
        with col2:
            st.metric("Dimension", stats["embeddings"]["embedding_dim"])
        with col3:
            st.metric("Device", stats["embeddings"]["device"].upper())
        with col4:
            st.metric("Batch Size", stats["embeddings"]["batch_size"])

        # LLM Configuration
        st.markdown("### 🤖 Language Model")
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Provider", stats["llm_provider"].upper())
        with col2:
            st.metric("Model", stats["llm_model"])

        # RAG Configuration
        st.markdown("### ⚙️ RAG Configuration")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Chunk Size", stats["chunk_size"])
        with col2:
            st.metric("Chunk Overlap", stats["chunk_overlap"])
        with col3:
            st.metric("Default Top-K", stats["retrieval_top_k"])

        # Raw JSON
        with st.expander("🔍 View Raw Stats (JSON)"):
            st.json(stats)

    else:
        st.error("Unable to fetch system statistics")
