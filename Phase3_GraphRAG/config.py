"""
config.py
=========
Phase 3 - Configurazione centralizzata.

Legge da variabili d'ambiente (python-dotenv opzionale).

Crea un file .env nella cartella Phase3_GraphRAG copiando .env.example
e riempi i valori prima di importare questo modulo.

Uso:
    from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE
    from config import LLM_MODEL, LLM_TEMPERATURE, ANTHROPIC_API_KEY, DATA_DIR
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
NEO4J_URI  = os.getenv("NEO4J_URI",  "bolt://127.0.0.1:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")

_neo4j_password_raw = os.getenv("NEO4J_PASSWORD", "")
if not _neo4j_password_raw:
    raise RuntimeError(
        "NEO4J_PASSWORD env var is not set. "
        "Copy Phase3_GraphRAG/.env.example to Phase3_GraphRAG/.env "
        "and fill in the password before importing config."
    )
NEO4J_PASSWORD: str = _neo4j_password_raw

NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

# --- LLM (Anthropic) ---
LLM_MODEL       = os.getenv("LLM_MODEL",       "claude-haiku-4-5-20251001")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))

_anthropic_key_raw = os.getenv("ANTHROPIC_API_KEY", "")
if not _anthropic_key_raw:
    raise RuntimeError(
        "ANTHROPIC_API_KEY env var is not set. "
        "Copy Phase3_GraphRAG/.env.example to Phase3_GraphRAG/.env "
        "and fill in your Anthropic API key before importing config."
    )
ANTHROPIC_API_KEY: str = _anthropic_key_raw

# --- Paths ---
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
