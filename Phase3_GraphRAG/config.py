"""
config.py
=========
Phase 3 - Configurazione centralizzata.

Legge da variabili d'ambiente se disponibili (python-dotenv opzionale),
altrimenti usa i valori di default per sviluppo locale.

Crea un file .env nella cartella Phase 3 copiando .env.example
per sovrascrivere i default senza modificare questo file.

Uso:
    from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE
    from config import LLM_MODEL, LLM_TEMPERATURE, DATA_DIR
"""

from __future__ import annotations

import os
from pathlib import Path

# Carica .env se presente (python-dotenv opzionale)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv non installato — usa variabili d'ambiente di sistema

# --- Neo4j ---
NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://127.0.0.1:7687")
NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "thesis2026")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

# --- LLM ---
LLM_MODEL       = os.getenv("LLM_MODEL",       "gpt-4o-mini")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))

# --- Paths ---
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
