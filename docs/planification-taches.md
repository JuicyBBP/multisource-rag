# Planification des Tâches - MultiSource RAG System

## 📅 Vue d'ensemble du projet

**Durée totale estimée** : 15-20 jours (à temps plein)
**Méthodologie** : Développement itératif avec MVP puis features avancées
**Priorité** : MVP fonctionnel d'abord, puis optimisations

---

## 🎯 Phases du projet

### Phase 0 : Setup & Préparation (Jour 1)
### Phase 1 : Backend Core (Jours 2-6)
### Phase 2 : Frontend (Jours 7-9)
### Phase 3 : Intégration & Tests (Jours 10-12)
### Phase 4 : Déploiement & Documentation (Jours 13-15)
### Phase 5 : Polish & Présentation (Jours 16-20)

---

## 📝 Détail des tâches

## PHASE 0 : Setup & Préparation

**Durée** : 1 jour
**Objectif** : Environnement de développement opérationnel

### Tâches

- [ ] **T0.1** : Structure du projet
  - Créer l'arborescence complète des dossiers
  - Initialiser Git repo
  - Setup `.gitignore` pour Python/ML
  - Créer branch `develop` et `main`
  - **Durée** : 30 min

- [ ] **T0.2** : Environment setup
  - Créer environnement virtuel Python 3.10+
  - Installer dependencies de base (voir requirements.txt)
  - Vérifier accès GPU (`nvidia-smi`)
  - Tester import de librairies clés
  - **Durée** : 1h

- [ ] **T0.3** : Configuration API
  - Obtenir clés API (OpenAI/Anthropic/Mistral)
  - Créer `.env.example` et `.env`
  - Tester appel API simple
  - Configurer rate limits si nécessaire
  - **Durée** : 30 min

- [ ] **T0.4** : Requirements & Dependencies
  - Créer `requirements.txt` avec versions fixées
  - Créer `requirements-dev.txt` (pytest, black, etc.)
  - Setup `pyproject.toml` pour tooling
  - **Durée** : 30 min

- [ ] **T0.5** : Configuration initiale
  - Créer `src/config.py` avec Pydantic Settings
  - Implémenter chargement des env vars
  - Ajouter validation de configuration
  - **Durée** : 1h

**Validation Phase 0** : ✅ Environnement fonctionnel, imports OK, API keys testées

---

## PHASE 1 : Backend Core

**Durée** : 5 jours
**Objectif** : Pipeline RAG fonctionnel avec API

### JOUR 2 : Document Loading & Processing

- [ ] **T1.1** : Service de chargement PDF
  - Implémenter `DocumentLoader` classe de base
  - Loader PDF avec PyPDF2 + pdfplumber (fallback)
  - Extraction de métadonnées (pages, titre)
  - Gestion des erreurs (PDF corrompus)
  - **Fichier** : `src/services/document_loader.py`
  - **Durée** : 3h

- [ ] **T1.2** : Loader Web Scraping
  - Scraper avec BeautifulSoup4
  - Extraction texte propre (sans scripts/styles)
  - Gestion des timeouts et erreurs HTTP
  - Support de redirections
  - **Durée** : 2h

- [ ] **T1.3** : Loader fichiers texte
  - Support .txt, .md, .docx
  - Détection automatique d'encodage
  - **Durée** : 1h

- [ ] **T1.4** : Text preprocessing
  - Cleaning (caractères spéciaux, espaces multiples)
  - Normalisation Unicode
  - Détection de langue (optionnel)
  - **Fichier** : `src/utils/text_processing.py`
  - **Durée** : 2h

**Tests Jour 2** : Charger 5 PDFs différents, 3 URLs, 2 fichiers texte

### JOUR 3 : Chunking & Embeddings

- [ ] **T1.5** : Implémentation du chunking
  - RecursiveCharacterTextSplitter
  - Configuration chunk_size=1000, overlap=200
  - Préservation des métadonnées par chunk
  - Tests avec différents types de documents
  - **Fichier** : `src/utils/text_processing.py`
  - **Durée** : 3h

- [ ] **T1.6** : Service d'embeddings
  - Charger modèle sentence-transformers
  - Fonction d'embedding batch (efficace)
  - Support GPU/CPU automatique
  - Caching des embeddings (optionnel)
  - **Fichier** : `src/services/embeddings.py`
  - **Durée** : 2h

- [ ] **T1.7** : Tests de performance embeddings
  - Mesurer vitesse GPU vs CPU
  - Tester batch sizes (16, 32, 64)
  - Vérifier VRAM usage
  - **Durée** : 1h

- [ ] **T1.8** : Notebook d'exploration
  - Comparer différents modèles d'embeddings
  - Visualiser similarité entre chunks
  - **Fichier** : `notebooks/01_exploration_embeddings.ipynb`
  - **Durée** : 2h

**Tests Jour 3** : Embeddings de 1000 chunks en <10 secondes

### JOUR 4 : Vector Store (ChromaDB)

- [ ] **T1.9** : Setup ChromaDB
  - Initialisation du client persistant
  - Création de collection avec métadonnées
  - Configuration de la distance metric (cosine)
  - **Fichier** : `src/models/database.py`
  - **Durée** : 2h

- [ ] **T1.10** : Service Vector Store
  - Méthode `add_documents(docs, embeddings, metadata)`
  - Méthode `search(query_embedding, top_k, filters)`
  - Méthode `delete_by_source(source_name)`
  - Méthode `get_stats()` (nombre de docs, sources)
  - **Fichier** : `src/services/vector_store.py`
  - **Durée** : 3h

- [ ] **T1.11** : Tests vector store
  - Insertion de 100 documents de test
  - Recherche et vérification des résultats
  - Tests de filtrage par métadonnées
  - Tests de suppression
  - **Fichier** : `tests/test_vector_store.py`
  - **Durée** : 2h

- [ ] **T1.12** : Persistence & Recovery
  - Vérifier persistence après redémarrage
  - Gestion de la corruption de DB
  - **Durée** : 1h

**Tests Jour 4** : DB avec 500+ chunks, recherches <100ms

### JOUR 5 : LLM Service & RAG Pipeline

- [ ] **T1.13** : Wrapper LLM API
  - Abstraction multi-providers (OpenAI, Anthropic, Mistral)
  - Gestion du streaming
  - Retry logic pour erreurs réseau
  - Rate limiting
  - Tracking des tokens/coûts
  - **Fichier** : `src/services/llm_service.py`
  - **Durée** : 3h

- [ ] **T1.14** : Prompt templates
  - Template système + user
  - Fonction de formatage du contexte
  - Gestion du dépassement de tokens
  - **Fichier** : `src/services/llm_service.py`
  - **Durée** : 1h

- [ ] **T1.15** : RAG Pipeline complet
  - Classe `RAGPipeline` orchestrant tout
  - Méthode `ingest_document(file_path, source_type)`
  - Méthode `query(question, top_k=5)`
  - Extraction des sources citées
  - **Fichier** : `src/services/rag_pipeline.py`
  - **Durée** : 3h

- [ ] **T1.16** : Tests end-to-end pipeline
  - Ingest 3 documents
  - Poser 10 questions de test
  - Vérifier qualité des réponses
  - Vérifier présence des citations
  - **Fichier** : `tests/test_rag_pipeline.py`
  - **Durée** : 1h

**Tests Jour 5** : Pipeline fonctionnel, réponses cohérentes avec sources

### JOUR 6 : API FastAPI

- [ ] **T1.17** : Setup FastAPI
  - App principale avec CORS
  - Configuration Uvicorn
  - Health check endpoint
  - **Fichier** : `src/main.py`
  - **Durée** : 1h

- [ ] **T1.18** : Pydantic schemas
  - `DocumentUploadRequest`
  - `QueryRequest`
  - `QueryResponse` (avec sources)
  - `DocumentInfo`
  - **Fichier** : `src/models/schemas.py`
  - **Durée** : 1h

- [ ] **T1.19** : Endpoints REST
  - `POST /api/documents/upload` (upload fichier)
  - `POST /api/documents/url` (ingest depuis URL)
  - `GET /api/documents` (liste des documents)
  - `DELETE /api/documents/{doc_id}`
  - `POST /api/query` (question/réponse)
  - `GET /api/stats` (statistiques)
  - **Fichier** : `src/api/routes.py`
  - **Durée** : 3h

- [ ] **T1.20** : WebSocket streaming
  - Endpoint `/ws/query`
  - Streaming des tokens de réponse
  - Gestion des connexions
  - **Fichier** : `src/api/websocket.py`
  - **Durée** : 2h

- [ ] **T1.21** : Tests API
  - Tests avec `TestClient`
  - Test de chaque endpoint
  - Test du WebSocket
  - **Fichier** : `tests/test_api.py`
  - **Durée** : 1h

**Validation Phase 1** : ✅ API fonctionnelle, pipeline RAG opérationnel, tests passent

---

## PHASE 2 : Frontend Streamlit

**Durée** : 3 jours
**Objectif** : Interface utilisateur élégante et intuitive

### JOUR 7 : Interface de base

- [ ] **T2.1** : Setup Streamlit
  - Configuration de base (`config.toml`)
  - Custom CSS pour styling professionnel
  - Layout avec sidebar
  - **Fichier** : `frontend/app.py`, `frontend/styles/custom.css`
  - **Durée** : 2h

- [ ] **T2.2** : Page d'accueil
  - Titre et description du projet
  - Instructions d'utilisation
  - Statistiques (nombre de docs, chunks)
  - **Durée** : 1h

- [ ] **T2.3** : Component Upload
  - File uploader avec drag & drop
  - Support multi-fichiers
  - Barre de progression upload
  - Prévisualisation du document uploadé
  - **Fichier** : `frontend/components/upload.py`
  - **Durée** : 3h

- [ ] **T2.4** : URL Ingestion
  - Input pour URL
  - Bouton "Importer depuis le web"
  - Feedback de succès/erreur
  - **Durée** : 1h

- [ ] **T2.5** : Liste des documents
  - Table avec documents ingérés
  - Colonnes : nom, type, date, taille
  - Boutons de suppression
  - **Durée** : 1h

**Tests Jour 7** : Upload fonctionne, documents apparaissent dans la liste

### JOUR 8 : Interface Chat

- [ ] **T2.6** : Component Chat
  - Container de messages (user/assistant)
  - Styling des bulles de chat
  - Timestamps
  - Avatars
  - **Fichier** : `frontend/components/chat.py`
  - **Durée** : 3h

- [ ] **T2.7** : Input utilisateur
  - `st.chat_input` pour questions
  - Désactivation pendant génération
  - Bouton "Nouvelle conversation"
  - **Durée** : 1h

- [ ] **T2.8** : Affichage des réponses
  - Streaming des tokens (avec WebSocket)
  - Markdown rendering
  - Code highlighting
  - **Durée** : 2h

- [ ] **T2.9** : Gestion de l'historique
  - Sauvegarde dans `st.session_state`
  - Scroll automatique vers bas
  - Export de conversation (JSON)
  - **Durée** : 2h

**Tests Jour 8** : Chat fonctionnel, historique persistant

### JOUR 9 : Sources & Polish

- [ ] **T2.10** : Component Sources
  - Affichage des sources citées
  - Expander par source
  - Affichage des extraits pertinents
  - Liens vers documents originaux
  - **Fichier** : `frontend/components/sources.py`
  - **Durée** : 3h

- [ ] **T2.11** : Feedback utilisateur
  - Boutons thumbs up/down par réponse
  - Enregistrement dans logs
  - **Durée** : 1h

- [ ] **T2.12** : Dashboard statistiques
  - Page séparée avec métriques
  - Graphiques Plotly (nombre de questions/jour, sources utilisées)
  - Coût total estimé
  - **Durée** : 2h

- [ ] **T2.13** : Polish UI
  - Peaufinage du CSS
  - Animations et transitions
  - Messages de chargement élégants
  - Dark mode (optionnel)
  - **Durée** : 2h

**Validation Phase 2** : ✅ Interface complète et professionnelle, UX fluide

---

## PHASE 3 : Intégration & Tests

**Durée** : 3 jours
**Objectif** : Système stable et testé

### JOUR 10 : Tests approfondis

- [ ] **T3.1** : Tests unitaires complets
  - Coverage de 80%+ sur services
  - Tests de edge cases
  - **Durée** : 3h

- [ ] **T3.2** : Tests d'intégration
  - Scénario complet : upload → query → response
  - Tests avec différents types de documents
  - Tests de charge (10+ requêtes simultanées)
  - **Durée** : 3h

- [ ] **T3.3** : Tests de qualité RAG
  - Créer dataset de Q&A de référence
  - Mesurer faithfulness et relevancy
  - Ajuster paramètres (top_k, threshold)
  - **Fichier** : `notebooks/03_evaluation_rag.ipynb`
  - **Durée** : 2h

**Tests Jour 10** : Tous les tests passent, coverage >80%

### JOUR 11 : Performance & Optimisation

- [ ] **T3.4** : Profiling
  - Identifier les bottlenecks
  - Mesurer latence de chaque composant
  - **Durée** : 2h

- [ ] **T3.5** : Optimisations
  - Batch processing pour embeddings
  - Caching des requêtes fréquentes (Redis optionnel)
  - Réduire taille des chunks si nécessaire
  - **Durée** : 3h

- [ ] **T3.6** : Tests de performance
  - Latence moyenne <3 secondes
  - VRAM usage <4GB
  - CPU usage raisonnable
  - **Durée** : 2h

- [ ] **T3.7** : Monitoring & Logging
  - Setup Loguru avec rotation
  - Logs structurés JSON
  - Tracking des métriques (latence, coût, erreurs)
  - **Fichier** : `src/utils/logger.py`, `src/utils/metrics.py`
  - **Durée** : 1h

**Tests Jour 11** : Performance acceptable, logs propres

### JOUR 12 : Debugging & Stabilisation

- [ ] **T3.8** : Gestion des erreurs
  - Try/except appropriés partout
  - Messages d'erreur clairs pour l'utilisateur
  - Retry logic robuste
  - **Durée** : 3h

- [ ] **T3.9** : Edge cases
  - Document vide
  - Question hors contexte
  - Très long document (>100 pages)
  - Caractères spéciaux/emojis
  - **Durée** : 2h

- [ ] **T3.10** : Documentation code
  - Docstrings pour toutes les fonctions
  - Type hints partout
  - Commentaires pour logique complexe
  - **Durée** : 2h

- [ ] **T3.11** : Code quality
  - Black formatting
  - Flake8 linting
  - MyPy type checking
  - **Durée** : 1h

**Validation Phase 3** : ✅ Système stable, performant, bien documenté

---

## PHASE 4 : Déploiement & Documentation

**Durée** : 3 jours
**Objectif** : Projet déployé et présentable

### JOUR 13 : Dockerization

- [ ] **T4.1** : Dockerfile backend
  - Multi-stage build
  - Optimisation des layers
  - Non-root user
  - **Fichier** : `docker/Dockerfile`
  - **Durée** : 2h

- [ ] **T4.2** : Dockerfile frontend
  - Image légère pour Streamlit
  - **Fichier** : `docker/Dockerfile.frontend`
  - **Durée** : 1h

- [ ] **T4.3** : Docker Compose
  - Services : backend, frontend, chromadb
  - Networks et volumes
  - Variables d'environnement
  - **Fichier** : `docker/docker-compose.yml`
  - **Durée** : 2h

- [ ] **T4.4** : Tests Docker
  - Build et run local
  - Vérifier communication entre services
  - **Durée** : 2h

**Tests Jour 13** : `docker-compose up` fonctionne parfaitement

### JOUR 14 : CI/CD & Déploiement

- [ ] **T4.5** : GitHub Actions - Tests
  - Workflow pour run tests sur push
  - Matrix testing (Python 3.10, 3.11)
  - Upload coverage reports
  - **Fichier** : `.github/workflows/tests.yml`
  - **Durée** : 2h

- [ ] **T4.6** : GitHub Actions - Deploy
  - Workflow pour deploy sur push main
  - Build et push Docker images
  - **Fichier** : `.github/workflows/deploy.yml`
  - **Durée** : 2h

- [ ] **T4.7** : Déploiement production
  - Choisir plateforme (Railway, Render, VPS)
  - Configurer domaine et HTTPS
  - Setup secrets en production
  - Déployer et tester
  - **Durée** : 3h

- [ ] **T4.8** : Monitoring production
  - Setup alertes (optionnel)
  - Vérifier logs
  - **Durée** : 1h

**Tests Jour 14** : Application accessible en ligne, CI/CD fonctionnel

### JOUR 15 : Documentation finale

- [ ] **T4.9** : README.md complet
  - Description du projet
  - Architecture diagram
  - Badges (CI status, coverage, license)
  - Features list
  - Quick start (3 commandes)
  - Screenshots/GIF
  - Technologies utilisées
  - Métriques de performance
  - Contribution guidelines
  - **Durée** : 3h

- [ ] **T4.10** : Documentation utilisateur
  - Guide d'utilisation
  - FAQ
  - Exemples de questions/réponses
  - **Fichier** : `docs/user-guide.md`
  - **Durée** : 2h

- [ ] **T4.11** : Documentation technique
  - Architecture détaillée
  - API documentation (Swagger)
  - Choix de design
  - **Fichier** : `docs/technical-details.md`
  - **Durée** : 2h

- [ ] **T4.12** : CHANGELOG.md
  - Versions et features
  - **Durée** : 30 min

**Validation Phase 4** : ✅ Projet déployé, documentation complète

---

## PHASE 5 : Polish & Présentation

**Durée** : 5 jours
**Objectif** : Portfolio-ready

### JOUR 16-17 : Polish & Features bonus

- [ ] **T5.1** : Features V2 (choisir 2-3)
  - [ ] Multi-query generation
  - [ ] Reranking avec cross-encoder
  - [ ] Export PDF de conversations
  - [ ] Recherche hybride (dense + sparse)
  - [ ] Support de plusieurs langues
  - **Durée** : 2 jours

- [ ] **T5.2** : UI/UX improvements
  - Animations
  - Meilleurs messages d'aide
  - Onboarding pour nouveaux users
  - **Durée** : 3h

- [ ] **T5.3** : Évaluation finale RAG
  - Dataset de test avec 20+ questions
  - Mesure de métriques
  - Rapport de qualité
  - **Durée** : 3h

**Tests Jours 16-17** : Features bonus fonctionnelles

### JOUR 18 : Contenu portfolio

- [ ] **T5.4** : Architecture diagram
  - Créer diagram propre (draw.io, excalidraw)
  - Montrer le flow complet
  - **Durée** : 2h

- [ ] **T5.5** : Screenshots & GIFs
  - Captures d'écran de l'UI
  - GIF de démonstration (upload + query)
  - Utiliser LICEcap ou Kap
  - **Durée** : 1h

- [ ] **T5.6** : Vidéo démo (optionnel mais puissant)
  - Script de 2-3 minutes
  - Enregistrement avec OBS/Loom
  - Montage simple
  - Upload sur YouTube
  - **Durée** : 3h

- [ ] **T5.7** : Blog post technique (optionnel)
  - Article Medium/Dev.to
  - Expliquer architecture et choix
  - Partager métriques et learnings
  - **Durée** : 2h

**Tests Jour 18** : Contenu visuel de qualité

### JOUR 19 : Préparation pitch

- [ ] **T5.8** : Pitch deck (5 slides)
  - Slide 1 : Problème & Solution
  - Slide 2 : Architecture technique
  - Slide 3 : Démo (screenshots)
  - Slide 4 : Métriques & Performance
  - Slide 5 : Technologies & Compétences
  - **Durée** : 3h

- [ ] **T5.9** : Préparer discours
  - Script de 2 minutes
  - Répéter présentation
  - Anticiper questions techniques
  - **Durée** : 2h

- [ ] **T5.10** : LinkedIn post
  - Annonce du projet
  - Highlights techniques
  - Lien vers GitHub et démo
  - **Durée** : 1h

**Tests Jour 19** : Pitch prêt, contenu marketing créé

### JOUR 20 : Revue finale

- [ ] **T5.11** : Revue de code
  - Lire tout le code avec œil critique
  - Nettoyer code mort
  - Améliorer nommage
  - **Durée** : 3h

- [ ] **T5.12** : Tests finaux
  - Scénarios utilisateur complets
  - Test sur différents navigateurs
  - Test de la démo publique
  - **Durée** : 2h

- [ ] **T5.13** : Checklist qualité
  - [ ] Tests passent (coverage >80%)
  - [ ] Pas de secrets dans le code
  - [ ] README complet
  - [ ] License présente
  - [ ] Démo en ligne accessible
  - [ ] CI/CD vert
  - [ ] Code formatté et linté
  - [ ] Documentation à jour
  - [ ] Architecture diagram présent
  - [ ] Screenshots/GIF dans README
  - **Durée** : 1h

- [ ] **T5.14** : Feedback externe
  - Montrer à 2-3 personnes
  - Recueillir critiques
  - Ajustements rapides
  - **Durée** : 2h

**Validation Phase 5** : ✅ Projet portfolio-ready, pitch préparé

---

## 📊 Tableau de suivi

| Phase | Tâches | Statut | Durée estimée | Durée réelle | Blockers |
|-------|--------|--------|---------------|--------------|----------|
| Phase 0 | 5 | ⬜ Pas commencé | 1 jour | - | - |
| Phase 1 | 21 | ⬜ Pas commencé | 5 jours | - | - |
| Phase 2 | 13 | ⬜ Pas commencé | 3 jours | - | - |
| Phase 3 | 11 | ⬜ Pas commencé | 3 jours | - | - |
| Phase 4 | 12 | ⬜ Pas commencé | 3 jours | - | - |
| Phase 5 | 14 | ⬜ Pas commencé | 5 jours | - | - |
| **TOTAL** | **76 tâches** | - | **20 jours** | - | - |

**Légende** :
- ⬜ Pas commencé
- 🟡 En cours
- ✅ Terminé
- ❌ Bloqué

---

## 🎯 Jalons (Milestones)

| Jalon | Critère de succès | Date cible |
|-------|-------------------|------------|
| M1 : Environment Ready | Setup complet, imports OK | Jour 1 |
| M2 : Backend MVP | Pipeline RAG fonctionnel | Jour 6 |
| M3 : Frontend MVP | Interface chat complète | Jour 9 |
| M4 : Tests OK | Coverage >80%, système stable | Jour 12 |
| M5 : Deployed | Application en ligne | Jour 14 |
| M6 : Portfolio-Ready | Documentation et démo parfaites | Jour 20 |

---

## ⚠️ Risques identifiés

| Risque | Probabilité | Impact | Mitigation |
|--------|-------------|--------|------------|
| API rate limits dépassés | Moyenne | Moyen | Implementer caching agressif, rate limiting côté client |
| Performance embeddings lente | Faible | Moyen | Batch processing, profiling early |
| Qualité RAG insuffisante | Moyenne | Élevé | Itérer sur chunking strategy, tester plusieurs top_k |
| Bugs ChromaDB | Faible | Élevé | Tests approfondis, backup strategy |
| Déploiement complexe | Moyenne | Moyen | Docker early, tester déploiement tôt |
| Scope creep | Élevée | Élevé | **Strict MVP first**, features V2 optionnelles |

---

## 💡 Conseils d'exécution

### Priorités absolues
1. **MVP d'abord** : Pipeline RAG simple mais fonctionnel avant features avancées
2. **Tests continus** : Ne pas accumuler de dette, tester au fur et à mesure
3. **Documentation en parallèle** : Écrire README au fur et à mesure, pas à la fin
4. **Deploy early** : Déployer dès le MVP pour éviter surprises

### Quand ajuster
- Si en retard : Supprimer features V2, focus sur MVP + démo
- Si en avance : Ajouter reranking, multi-query, ou meilleure UI
- Si bloqué : Documenter le blocker, chercher aide, ou skip temporairement

### Suivi quotidien
- [ ] Début de journée : Revoir tâches du jour
- [ ] Fin de journée : Mettre à jour statuts, noter blockers
- [ ] Commit code au moins 1x par jour avec message clair
- [ ] Tester manuellement ce qui a été développé

---

## 📈 Métriques de succès du projet

### Techniques
- ✅ Tests coverage >80%
- ✅ Latence moyenne <3s par requête
- ✅ 0 erreurs en production sur 100 queries
- ✅ Support de 3+ types de documents
- ✅ API documentée (Swagger)

### Portfolio
- ✅ Démo en ligne accessible 24/7
- ✅ README avec >5 sections complètes
- ✅ Architecture diagram professionnel
- ✅ GIF de démo dans README
- ✅ 100+ commits avec historique propre

### Impact recrutement
- ✅ Projet cité en entretien
- ✅ Questions techniques des recruteurs
- ✅ Démontre 6+ compétences clés
- ✅ Différenciation vs autres candidats

---

## 🔄 Template de rapport quotidien

```markdown
## Jour X - [Date]

### Tâches complétées
- [x] T1.1 : Description
- [x] T1.2 : Description

### Tâches en cours
- [ ] T1.3 : Description (50% done)

### Blockers
- Problème avec ChromaDB persistence → chercher docs

### Learnings
- TIL : Les embeddings batch sont 5x plus rapides

### Demain
- Focus : Terminer T1.3 et T1.4
- Objectif : Avoir embeddings fonctionnels
```

---

## ✅ Checklist de fin de projet

### Code
- [ ] Tous les tests passent
- [ ] Coverage >80%
- [ ] Code formatté (Black)
- [ ] Type hints partout
- [ ] Pas de code mort
- [ ] Pas de secrets exposés
- [ ] .gitignore complet

### Documentation
- [ ] README complet avec badges
- [ ] Architecture diagram
- [ ] API docs (Swagger)
- [ ] CHANGELOG.md
- [ ] LICENSE (MIT recommandé)
- [ ] Docstrings dans code

### Déploiement
- [ ] Application déployée et accessible
- [ ] HTTPS configuré
- [ ] CI/CD opérationnel (vert)
- [ ] Monitoring basique en place
- [ ] Backup de la DB

### Portfolio
- [ ] Screenshots de qualité
- [ ] GIF de démo (<5MB)
- [ ] Vidéo démo (optionnel)
- [ ] Pitch deck prêt
- [ ] LinkedIn post publié

### Préparation entretien
- [ ] Capable d'expliquer chaque choix technique
- [ ] Connaître les métriques par cœur
- [ ] Scénarios de scale-up préparés
- [ ] Améliorations futures identifiées
- [ ] Capable de live debug le code

---

**Version** : 1.0
**Dernière mise à jour** : 2025-12-09
**Prochaine revue** : À la fin de chaque phase

**Note** : Cette planification est un guide, pas une contrainte. Adapter selon les découvertes et blockers rencontrés.
