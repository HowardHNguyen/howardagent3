import os
"""Loading LLMs and Embeddings."""

from config import set_environment

set_environment()

# Ensure local cache directory exists (Streamlit Cloud uses ephemeral FS)
os.makedirs("./cache", exist_ok=True)

# --- LangChain imports (version-safe) ---
try:
    # Newer LangChain locations
    from langchain.embeddings.cache import CacheBackedEmbeddings
except Exception:
    # Older LangChain fallback
    from langchain.embeddings import CacheBackedEmbeddings

try:
    from langchain.storage import LocalFileStore
except Exception:
    # Older fallback (rare)
    from langchain.storage import LocalFileStore

from langchain_openai import OpenAIEmbeddings
from langchain_groq import ChatGroq

chat_model = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

store = LocalFileStore("./cache/")

underlying_embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
)

# Cache embeddings to avoid repeat costs
EMBEDDINGS = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings,
    store,
    namespace=underlying_embeddings.model,
)
