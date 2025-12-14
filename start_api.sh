#!/bin/bash
# Script to start the FastAPI server

echo "🚀 Starting FastAPI server..."
echo "API will be available at: http://localhost:8000"
echo "API docs will be available at: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

export PYTHONPATH=/mnt/e/projetIA
.venv/bin/python src/api/main.py
