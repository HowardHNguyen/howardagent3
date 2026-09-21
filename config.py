"""
Environment configuration helper.

This app can run locally or on Streamlit Cloud.

Local:
  - export GROQ_API_KEY=...
  - export OPENAI_API_KEY=...

Streamlit Cloud:
  - add secrets in the app settings:
      GROQ_API_KEY = "..."
      OPENAI_API_KEY = "..."
"""
from __future__ import annotations

import os


def set_environment() -> None:
    """
    Ensure required API keys are available in os.environ.
    Tries (in order):
      1) existing environment variables
      2) Streamlit secrets (if running under Streamlit)
    """
    try:
        import streamlit as st  # type: ignore
        secrets = getattr(st, "secrets", None)
    except Exception:
        secrets = None

    def _set_from_secrets(key: str) -> None:
        if os.environ.get(key):
            return
        if secrets is None:
            return
        try:
            val = secrets.get(key)
        except Exception:
            val = None
        if val:
            os.environ[key] = str(val)

    _set_from_secrets("GROQ_API_KEY")
    _set_from_secrets("OPENAI_API_KEY")
