# Spécifications Techniques - MultiSource RAG System

## 📋 Vue d'ensemble du projet

### Objectif
Développer un système RAG (Retrieval-Augmented Generation) avancé capable d'interroger intelligemment plusieurs sources de données (documents PDF, pages web, bases de données) pour fournir des réponses contextuelles précises avec citations.

### Proposition de valeur
- **Pour les recruteurs** : Démontre la maîtrise des LLMs, architectures modernes, et capacité à créer des systèmes production-ready
- **Différenciation** : Multi-sources, interface élégante, déploiement complet avec monitoring
- **Complexité technique** : Architecture modulaire, optimisations GPU, gestion de contexte intelligente

---

## 🎯 Fonctionnalités principales

### MVP (Version 1.0)
1. **Ingestion multi-sources**
   - Upload et parsing de documents PDF
   - Scraping de pages web via URL
   - Support de fichiers texte (.txt, .md)

2. **Système RAG de base**
   - Chunking intelligent des documents
   - Embeddings avec modèles optimisés
   - Recherche vectorielle avec ChromaDB
   - Génération de réponses avec citations

3. **Interface utilisateur**
   - Chat interface avec Streamlit
   - Historique de conversation
   - Affichage des sources utilisées
   - Upload drag & drop

4. **Backend API**
   - FastAPI avec endpoints REST
   - WebSocket pour streaming des réponses
   - Gestion de sessions utilisateur

### Features avancées (Version 2.0)
- Multi-query generation pour améliorer le recall
- Reranking des résultats avec Cross-Encoder
- Connexion à bases de données (PostgreSQL)
- Recherche hybride (vectorielle + BM25)
- Cache intelligent des requêtes fréquentes
- Export de conversations en PDF
- Comparaison de sources contradictoires

---

## 🏗️ Architecture technique

### Stack technologique

#### Backend
- **Framework** : FastAPI 0.109+
- **LLM Provider** : OpenAI API / Anthropic Claude / Mistral API (configurable)
- **Vector Database** : ChromaDB 0.4.22+
- **Embeddings** :
  - Primaire : `sentence-transformers/all-MiniLM-L6-v2` (léger, 384 dim)
  - Alternative : `BAAI/bge-small-en-v1.5` (meilleure qualité)
- **Document Processing** :
  - PyPDF2 / pdfplumber pour PDFs
  - BeautifulSoup4 + Playwright pour web scraping
  - python-docx pour Word
- **Cache** : Redis (optionnel pour production)

#### Frontend
- **Framework** : Streamlit 1.31+
- **Styling** : Custom CSS pour look professionnel
- **Charts** : Plotly pour visualisations

#### MLOps
- **Containerization** : Docker + Docker Compose
- **Monitoring** :
  - Logging structuré avec Loguru
  - Métriques custom (latence, coût tokens, satisfaction)
  - Prometheus + Grafana (optionnel)
- **Testing** : pytest + pytest-cov
- **CI/CD** : GitHub Actions

---

## 📊 Architecture des données

### Pipeline d'ingestion

```
Document Source → Loader → Text Splitter → Embeddings → Vector Store
                                  ↓
                            Metadata Extraction
                         (source, page, timestamp)
```

#### Stratégie de chunking
- **Méthode** : RecursiveCharacterTextSplitter
- **Chunk size** : 1000 caractères
- **Overlap** : 200 caractères (20%)
- **Séparateurs** : `["\n\n", "\n", ". ", " ", ""]`

#### Métadonnées stockées
```json
{
  "source": "nom_fichier.pdf",
  "source_type": "pdf|web|txt|db",
  "page": 5,
  "chunk_id": "uuid",
  "timestamp": "2025-12-09T10:30:00Z",
  "char_count": 987,
  "url": "https://..." // si web
}
```

### Structure de la base vectorielle

**Collection ChromaDB** : `documents_collection`
- **Vecteurs** : Embeddings 384-dim (MiniLM) ou 768-dim (BGE)
- **Distance metric** : Cosine similarity
- **Index** : HNSW pour recherche rapide

---

## 🔄 Flow de requête RAG

### Étapes du pipeline

1. **Réception de la question utilisateur**
   - Validation et sanitization
   - Détection de la langue (pour support multilingue futur)

2. **Query Enhancement** (optionnel v2.0)
   - Reformulation de la question
   - Génération de queries multiples
   - Expansion avec synonymes

3. **Retrieval**
   ```python
   # Paramètres de recherche
   top_k = 5  # Nombre de chunks récupérés
   similarity_threshold = 0.7  # Seuil de pertinence
   ```
   - Embedding de la question
   - Recherche vectorielle dans ChromaDB
   - Filtrage par score de similarité

4. **Reranking** (optionnel v2.0)
   - Cross-encoder pour reordonner les résultats
   - Modèle : `cross-encoder/ms-marco-MiniLM-L-6-v2`

5. **Context Building**
   - Agrégation des chunks pertinents
   - Déduplication des sources
   - Formatage avec métadonnées

6. **Generation**
   - Construction du prompt avec contexte
   - Appel API LLM (streaming mode)
   - Post-processing de la réponse

7. **Citation & Source Tracking**
   - Extraction des sources utilisées
   - Mapping vers documents originaux
   - Affichage des extraits pertinents

---

## 🎨 Design du Prompt

### Template de base

```python
SYSTEM_PROMPT = """Tu es un assistant expert qui répond aux questions en te basant
UNIQUEMENT sur les documents fournis.

Règles importantes :
1. Cite toujours tes sources en utilisant [Source: nom_fichier, page X]
2. Si l'information n'est pas dans les documents, dis "Je ne trouve pas cette
   information dans les documents fournis"
3. Sois précis et concis
4. Si plusieurs sources se contredisent, mentionne-le explicitement
"""

USER_PROMPT = """
Contexte fourni :
{context}

Question : {question}

Réponds en français de manière claire et structurée.
"""
```

### Gestion du contexte
- **Max tokens pour contexte** : 4000 tokens (~3000 mots)
- **Stratégie si dépassement** :
  - Prioriser les chunks avec meilleur score
  - Truncation intelligente sur phrases complètes

---

## 🔧 Configuration système

### Requirements GPU
- **Embeddings** : ~500MB VRAM (all-MiniLM-L6-v2 en local)
- **Alternative** : Utiliser API d'embeddings (OpenAI, Cohere) pour libérer GPU
- **Recommandation** : RTX 3070 8GB largement suffisante

### Requirements compute
```yaml
CPU: 4+ cores
RAM: 8GB minimum, 16GB recommandé
Storage: 5GB pour environnement + documents
GPU: Optionnel (embeddings possibles sur CPU)
```

### Variables d'environnement
```bash
# LLM API
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-...
MISTRAL_API_KEY=...

# LLM Config
LLM_PROVIDER=openai  # openai|anthropic|mistral
LLM_MODEL=gpt-4-turbo-preview
LLM_TEMPERATURE=0.1
LLM_MAX_TOKENS=1500

# Embeddings
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DEVICE=cuda  # cuda|cpu

# ChromaDB
CHROMA_PERSIST_DIRECTORY=./data/chroma_db
CHROMA_COLLECTION_NAME=documents_collection

# App Config
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
RETRIEVAL_TOP_K=5
SIMILARITY_THRESHOLD=0.7

# Redis (optionnel)
REDIS_HOST=localhost
REDIS_PORT=6379
```

---

## 📁 Structure du projet

```
MultiSource-RAG-System/
├── docs/
│   ├── specifications-techniques.md     # Ce fichier
│   ├── planification-taches.md
│   └── architecture-diagram.png         # À créer
├── src/
│   ├── __init__.py
│   ├── main.py                          # Point d'entrée FastAPI
│   ├── config.py                        # Configuration & env vars
│   ├── models/
│   │   ├── __init__.py
│   │   ├── schemas.py                   # Pydantic models
│   │   └── database.py                  # ChromaDB setup
│   ├── services/
│   │   ├── __init__.py
│   │   ├── document_loader.py           # Ingestion multi-sources
│   │   ├── embeddings.py                # Embeddings service
│   │   ├── vector_store.py              # ChromaDB operations
│   │   ├── llm_service.py               # LLM API wrapper
│   │   └── rag_pipeline.py              # Pipeline complet
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes.py                    # Endpoints REST
│   │   └── websocket.py                 # Streaming responses
│   └── utils/
│       ├── __init__.py
│       ├── text_processing.py           # Chunking, cleaning
│       ├── logger.py                    # Logging setup
│       └── metrics.py                   # Tracking métriques
├── frontend/
│   ├── app.py                           # Streamlit app
│   ├── components/
│   │   ├── chat.py                      # Interface chat
│   │   ├── upload.py                    # Upload de documents
│   │   └── sources.py                   # Affichage sources
│   └── styles/
│       └── custom.css                   # Styling
├── tests/
│   ├── __init__.py
│   ├── test_document_loader.py
│   ├── test_embeddings.py
│   ├── test_rag_pipeline.py
│   └── test_api.py
├── data/
│   ├── chroma_db/                       # Vector DB persistence
│   ├── uploaded_docs/                   # Documents uploadés
│   └── cache/                           # Query cache
├── notebooks/
│   ├── 01_exploration_embeddings.ipynb
│   ├── 02_chunking_strategies.ipynb
│   └── 03_evaluation_rag.ipynb
├── docker/
│   ├── Dockerfile
│   ├── Dockerfile.frontend
│   └── docker-compose.yml
├── .github/
│   └── workflows/
│       ├── tests.yml
│       └── deploy.yml
├── .env.example
├── .gitignore
├── requirements.txt
├── requirements-dev.txt
├── README.md
├── setup.py
└── pytest.ini
```

---

## 🧪 Stratégie de test

### Tests unitaires
- Coverage minimum : 80%
- Tests pour chaque service indépendamment
- Mock des appels API externes

### Tests d'intégration
- Pipeline RAG end-to-end
- Endpoints API avec TestClient
- Simulation de documents de test

### Tests de performance
- Latence de requête < 3 secondes
- Throughput : 10+ requêtes/seconde
- Memory footprint < 4GB

### Métriques de qualité RAG
- **Faithfulness** : La réponse est-elle fidèle au contexte ?
- **Answer Relevancy** : La réponse répond-elle à la question ?
- **Context Precision** : Les chunks récupérés sont-ils pertinents ?
- **Context Recall** : Tous les chunks nécessaires sont-ils récupérés ?

---

## 🚀 Plan de déploiement

### Environnements

1. **Development**
   - Local avec hot reload
   - Base ChromaDB en mémoire
   - Logs verbose

2. **Staging** (optionnel)
   - Docker Compose
   - ChromaDB persistente
   - Tests d'intégration automatiques

3. **Production**
   - **Option 1** : Docker Compose sur VPS
   - **Option 2** : Kubernetes (overkill pour portfolio)
   - **Option 3** : Railway / Render / Fly.io
   - HTTPS avec Let's Encrypt
   - Monitoring activé

### Déploiement du frontend
- Streamlit Community Cloud (gratuit)
- Ou avec le backend sur même serveur

---

## 📈 Monitoring & Observability

### Logs structurés
```python
{
  "timestamp": "2025-12-09T10:30:00Z",
  "level": "INFO",
  "service": "rag_pipeline",
  "event": "query_processed",
  "user_id": "abc123",
  "question": "...",
  "retrieved_docs": 5,
  "generation_time": 2.3,
  "total_tokens": 1200,
  "cost_usd": 0.0024
}
```

### Métriques à tracker
- Nombre de requêtes / jour
- Latence moyenne par composant
- Coût API par requête
- Taux d'erreur
- Distribution des sources utilisées
- Satisfaction utilisateur (feedback thumbs up/down)

### Dashboard Grafana (optionnel)
- Graphiques de latence
- Coût cumulé
- Volume de documents
- Taux de cache hit

---

## 🔒 Sécurité

### Considérations
1. **API Keys** : Jamais en dur, toujours via .env
2. **Rate limiting** : 100 requêtes/heure par IP
3. **Input validation** : Sanitization des uploads
4. **File size limits** : Max 10MB par document
5. **Content moderation** : Filtrage des prompts malveillants
6. **CORS** : Configuration restrictive en production

### Secrets management
- `.env` pour local
- Secrets manager en production (AWS Secrets, Doppler)

---

## 💰 Estimation des coûts

### Coûts d'API (par 1000 requêtes)

**Embeddings** (si API externe)
- OpenAI `text-embedding-3-small` : ~$0.02/1M tokens → négligeable
- En local (gratuit) : All-MiniLM-L6-v2

**LLM Generation**
- GPT-4 Turbo : $0.01/1K input tokens, $0.03/1K output → ~$0.05/requête
- Claude Sonnet 3.5 : $0.003/1K input, $0.015/1K output → ~$0.025/requête
- Mistral Small : ~$0.002/1K tokens → ~$0.01/requête

**Estimation réaliste pour démo portfolio** : $5-10/mois avec usage modéré

---

## 📚 Documentation pour le portfolio

### README.md à inclure
- Badge CI/CD status
- Démo GIF/vidéo
- Architecture diagram
- Instructions d'installation en 3 commandes
- Exemples de questions/réponses
- Métriques de performance
- Technologies utilisées avec badges

### Éléments impressionnants à montrer
1. **Architecture claire** : Diagram avec draw.io
2. **Métriques concrètes** : "Répond en moyenne en 2.3s avec précision de 87%"
3. **Démo live** : URL fonctionnelle à tester
4. **Code quality** : Tests, linting, type hints
5. **Production-ready** : Docker, monitoring, CI/CD

---

## 🎓 Compétences démontrées

✅ **LLMs & RAG** : Architecture moderne, prompt engineering
✅ **Vector Databases** : ChromaDB, embeddings, similarity search
✅ **APIs** : FastAPI, WebSockets, REST design
✅ **Frontend** : Streamlit, UX design
✅ **MLOps** : Docker, CI/CD, monitoring
✅ **Software Engineering** : Architecture modulaire, tests, documentation
✅ **Optimisation** : Gestion GPU, caching, performance

---

## 📋 Checklist finale avant présentation

- [ ] Code propre avec type hints et docstrings
- [ ] Tests avec coverage > 80%
- [ ] Documentation README complète avec exemples
- [ ] Démo déployée et accessible en ligne
- [ ] Architecture diagram professionnel
- [ ] Métriques de performance mesurées
- [ ] Code sur GitHub avec historique de commits propre
- [ ] License open-source (MIT)
- [ ] CHANGELOG.md avec versions
- [ ] Vidéo démo de 2-3 minutes (optionnel mais fort impact)

---

**Version** : 1.0
**Dernière mise à jour** : 2025-12-09
**Auteur** : [Votre nom]
**Complexité estimée** : Intermédiaire à Avancé
**Temps de développement estimé** : 2-3 semaines à temps plein
