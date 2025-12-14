# Guide de Test - MultiSource RAG System

## 📋 Prérequis

Assurez-vous que :
- ✅ L'environnement virtuel est activé
- ✅ Toutes les dépendances sont installées
- ✅ Le fichier `.env` contient votre clé API Mistral (ou OpenAI/Anthropic)

## 🚀 Méthode 1 : Test Complet avec Interface Web (Recommandé)

### Étape 1 : Lancer l'API Backend

Ouvrez un premier terminal et exécutez :

```bash
./start_api.sh
```

Vous devriez voir :
```
🚀 Starting FastAPI server...
API will be available at: http://localhost:8000
```

Attendez que le message apparaisse :
```
INFO: Application startup complete.
INFO: Uvicorn running on http://0.0.0.0:8000
```

### Étape 2 : Lancer le Frontend Streamlit

Ouvrez un **deuxième terminal** et exécutez :

```bash
./start_frontend.sh
```

Vous devriez voir :
```
🎨 Starting Streamlit frontend...
Frontend will be available at: http://localhost:8501
```

Le navigateur devrait s'ouvrir automatiquement à `http://localhost:8501`

### Étape 3 : Tester l'Upload de Documents

1. Dans l'interface Streamlit, allez sur **📤 Upload Documents**
2. Cliquez sur **Browse files**
3. Sélectionnez un document de test (PDF, DOCX, TXT ou MD)
4. Cliquez sur **🚀 Upload and Process**
5. Attendez que le traitement se termine (quelques secondes)

Vous devriez voir : `✅ votre_document.ext: X chunks created`

### Étape 4 : Tester les Questions/Réponses

1. Allez sur **💬 Ask Questions**
2. Posez une question dans le chat, par exemple :
   - "Quels sont les trois types de machine learning ?"
   - "Qu'est-ce que le deep learning ?"
   - "Quelles sont les applications du machine learning ?"

3. Observez :
   - La réponse générée par l'IA
   - Les sources utilisées (avec score de similarité)
   - Le contexte extrait de vos documents

### Étape 5 : Consulter les Statistiques

1. Allez sur **📊 Statistics**
2. Vérifiez :
   - Nombre de chunks stockés
   - Nombre de documents
   - Configuration du modèle d'embeddings
   - Paramètres du LLM

---

## 🔧 Méthode 2 : Test via l'API Directement

### Test 1 : Health Check

```bash
curl http://localhost:8000/api/v1/health
```

Réponse attendue :
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "components": {...}
}
```

### Test 2 : Upload d'un Document

```bash
curl -X POST "http://localhost:8000/api/v1/ingest" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@votre_document.txt"
```

### Test 3 : Poser une Question

```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Quels sont les types de machine learning ?",
    "n_results": 5,
    "min_similarity": 0.7,
    "include_sources": true
  }'
```

### Test 4 : Obtenir les Statistiques

```bash
curl http://localhost:8000/api/v1/stats
```

---

## 🧪 Méthode 3 : Test avec Script Python

Créez un fichier `test_api.py` :

```python
import requests
import json

API_URL = "http://localhost:8000/api/v1"

# 1. Test health
print("1. Testing health endpoint...")
response = requests.get(f"{API_URL}/health")
print(f"Status: {response.status_code}")
print(f"Response: {response.json()}\n")

# 2. Upload document
print("2. Uploading document...")
with open("votre_document.txt", "rb") as f:
    files = {"file": ("votre_document.txt", f, "text/plain")}
    response = requests.post(f"{API_URL}/ingest", files=files)
print(f"Status: {response.status_code}")
print(f"Response: {json.dumps(response.json(), indent=2)}\n")

# 3. Query
print("3. Querying RAG system...")
query_data = {
    "question": "Quels sont les trois types de machine learning ?",
    "n_results": 3,
    "min_similarity": 0.7,
    "include_sources": True
}
response = requests.post(f"{API_URL}/query", json=query_data)
print(f"Status: {response.status_code}")
result = response.json()
print(f"Answer: {result['answer']}\n")
print(f"Number of sources: {result['num_sources']}\n")

# 4. Stats
print("4. Getting statistics...")
response = requests.get(f"{API_URL}/stats")
print(f"Total chunks: {response.json()['vector_store']['total_chunks']}")
```

Puis exécutez :
```bash
PYTHONPATH=/mnt/e/projetIA .venv/bin/python test_api.py
```

---

## 📝 Conseils pour les Questions

Après avoir uploadé vos documents, testez différents types de questions :

1. **Questions factuelles** (réponse directe dans le texte)
   - Demandez des définitions, listes ou faits spécifiques

2. **Questions de compréhension**
   - Demandez des comparaisons ou explications

3. **Questions de synthèse**
   - Demandez des résumés ou des vues d'ensemble

---

## 🐛 Dépannage

### L'API ne démarre pas

**Problème** : `ModuleNotFoundError`
```bash
# Solution : Vérifier le PYTHONPATH
export PYTHONPATH=/mnt/e/projetIA
```

**Problème** : `Error loading LLM client`
```bash
# Solution : Vérifier votre clé API dans .env
cat .env | grep MISTRAL_API_KEY
```

### Le Frontend ne se connecte pas à l'API

**Problème** : "❌ API Offline" dans Streamlit

**Solutions** :
1. Vérifiez que l'API est bien lancée sur le port 8000
2. Testez avec : `curl http://localhost:8000/api/v1/health`
3. Vérifiez qu'il n'y a pas de firewall bloquant le port 8000

### Les réponses sont incohérentes

**Solutions** :
1. Vérifiez dans Statistics que des chunks sont bien stockés
2. Augmentez le nombre de sources (n_results) dans les paramètres
3. Diminuez le seuil de similarité minimum

### GPU non détecté

**Problème** : "CUDA not available"

**Solutions** :
1. Vérifiez avec : `.venv/bin/python -c "import torch; print(torch.cuda.is_available())"`
2. Si False, le système utilisera le CPU (plus lent mais fonctionnel)

---

## ✅ Checklist de Test

- [ ] API démarre sans erreur
- [ ] Frontend se connecte à l'API
- [ ] Upload d'un document réussit
- [ ] Chunks sont créés et stockés
- [ ] Question simple obtient une réponse
- [ ] Sources sont affichées avec scores
- [ ] Statistiques affichent les bonnes valeurs
- [ ] Multiple questions gardent l'historique

---

## 🎯 Test Rapide (2 minutes)

```bash
# Terminal 1
./start_api.sh

# Terminal 2 (nouveau terminal)
./start_frontend.sh

# Dans le navigateur (http://localhost:8501)
# 1. Upload votre premier document (PDF, DOCX, TXT)
# 2. Posez une question sur le contenu
# 3. Vérifiez la réponse et les sources
```

Si tout fonctionne, vous êtes prêt à utiliser le système ! 🎉
