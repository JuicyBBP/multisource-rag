#!/bin/bash
# Script to start the Streamlit frontend

echo "🎨 Starting Streamlit frontend..."
echo "Frontend will be available at: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

export PYTHONPATH=/mnt/e/projetIA
.venv/bin/python -m streamlit run src/frontend/app.py \
  --server.address=0.0.0.0 \
  --server.port=8501 \
  --server.headless=true
