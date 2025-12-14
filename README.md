# MultiSource RAG System

> Système RAG (Retrieval-Augmented Generation) avancé permettant d'interroger intelligemment plusieurs sources de données (PDF, Web, Texte) avec interface chat élégante.

## Vue d'ensemble

Ce projet est un système RAG production-ready qui démontre :
- Architecture LLM moderne avec pipeline RAG complet
- Ingestion multi-sources (PDF, URLs, fichiers texte)
- Recherche vectorielle avec ChromaDB
- API REST avec FastAPI et WebSocket streaming
- Interface utilisateur intuitive avec Streamlit
- MLOps : Docker, CI/CD, monitoring, tests

## Technologies

**Backend**
- FastAPI + Uvicorn
- ChromaDB (vector database)
- Sentence Transformers (embeddings)
- LangChain (orchestration)
- OpenAI / Anthropic / Mistral API (LLM)

**Frontend**
- Streamlit
- Plotly (visualisations)

**MLOps**
- Docker + Docker Compose
- pytest (tests, coverage >80%)
- GitHub Actions (CI/CD)
- Loguru (logging structuré)

## Quick Start

```bash
# 1. Clone et setup
git clone <repo-url>
cd MultiSource-RAG-System
python -m venv venv
source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt

# 2. Configuration
cp .env.example .env
# Éditer .env avec vos API keys

# 3. Lancer l'application
docker-compose up
```

Accédez à :
- Frontend : http://localhost:8501
- API : http://localhost:8000
- API Docs : http://localhost:8000/docs

## Documentation

- [Spécifications Techniques](docs/specifications-techniques.md) - Architecture détaillée, stack tech, design

## Fonctionnalités

- Upload de documents PDF, Word, texte
- Import de contenu web via URL
- Chat interface avec historique
- Réponses avec citations et sources
- Recherche vectorielle optimisée
- Streaming des réponses en temps réel
- Gestion multi-utilisateurs
- Dashboard de statistiques
- Export de conversations

## Architecture

```
User → Streamlit UI → FastAPI Backend → RAG Pipeline
                                          ├─ Document Loader
                                          ├─ Text Chunker
                                          ├─ Embeddings (GPU)
                                          ├─ ChromaDB (vectors)
                                          └─ LLM API
```

## Requirements

- Python 3.10+
- CUDA-capable GPU (optionnel, améliore les embeddings)
- 8GB RAM minimum
- Clés API : OpenAI, Anthropic, ou Mistral

## Développement

```bash
# Installation dev dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/ -v --cov=src --cov-report=html

# Format code
black src/ tests/
flake8 src/ tests/

# Type checking
mypy src/
```

## Métriques

- Tests coverage : >80%
- Latence moyenne : <3s par requête
- Support : 3+ types de documents
- Architecture : Modulaire avec 6+ services
