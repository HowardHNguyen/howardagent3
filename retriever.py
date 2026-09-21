"""Retriever module (Pydantic-safe for LangChain BaseRetriever)."""

import os
import tempfile
from typing import List, Any

from openai import APIConnectionError, AuthenticationError, RateLimitError
from pydantic.v1 import Field

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from document_loader import load_document
from llms import EMBEDDINGS

# One in-memory vector store for the app session
VECTOR_STORE = InMemoryVectorStore(embedding=EMBEDDINGS)


class EmbeddingServiceError(Exception):
    """Raised when the embeddings API call fails for a user-facing reason
    (out of credit, bad/expired key, or temporary connection issue)."""


def _run_embedding_call(fn, *args, **kwargs):
    """Run an embeddings-related call, translating OpenAI errors into a
    single friendly EmbeddingServiceError with actionable guidance."""
    try:
        return fn(*args, **kwargs)
    except RateLimitError as e:
        msg = str(e).lower()
        if "quota" in msg or "billing" in msg or "credit" in msg:
            raise EmbeddingServiceError(
                "The embeddings service reported that the OpenAI account has "
                "run out of credit or hit its billing limit. Please add credit "
                "at platform.openai.com (Settings -> Billing) and try again."
            ) from e
        raise EmbeddingServiceError(
            "The embeddings service is temporarily rate-limited (too many "
            "requests). Please wait a minute and try again."
        ) from e
    except AuthenticationError as e:
        raise EmbeddingServiceError(
            "The OpenAI API key is missing, invalid, or has been revoked. "
            "Please check the OPENAI_API_KEY value in your app secrets."
        ) from e
    except APIConnectionError as e:
        raise EmbeddingServiceError(
            "Could not reach the OpenAI API (network/connection issue). "
            "Please try again in a moment."
        ) from e


class DocumentRetriever(BaseRetriever):
    """Stores documents in an in-memory vector store and retrieves by similarity search."""

    k: int = 4
    documents: List[Document] = Field(default_factory=list)

    def store_documents(self, docs: List[Document]) -> None:
        """Split and add docs to the vector store."""
        if not docs:
            return

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = splitter.split_documents(docs)

        _run_embedding_call(VECTOR_STORE.add_documents, split_docs)

    def add_documents_from_uploads(self, uploaded_files: List[Any]) -> None:
        """Load Streamlit uploaded files and add them to the vector store."""
        docs: List[Document] = []

        for file in uploaded_files:
            suffix = os.path.splitext(file.name)[-1]

            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(file.getbuffer())
                temp_filepath = tmp.name

            try:
                file_docs = load_document(temp_filepath)
                docs.extend(file_docs)
            except Exception as e:
                # Keep app running even if one doc fails
                print(f"Failed to load {file.name}: {e}")
            finally:
                try:
                    os.remove(temp_filepath)
                except Exception:
                    pass

        # Update stored docs and vector store
        if docs:
            self.documents.extend(docs)
            self.store_documents(docs)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        """Retrieve relevant chunks from the vector store."""
        if not self.documents:
            return []
        return _run_embedding_call(
            VECTOR_STORE.similarity_search, query=query, k=self.k
        )
