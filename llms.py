"""Lazy, bounded provider calls. No disk cache or raw provider error output."""
import logging
import groq
import openai
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from config import Settings

LOGGER = logging.getLogger(__name__)


class ServiceError(Exception):
    """A deliberately sanitized error safe to display in the public app."""


def provider_call(provider, operation, *args, **kwargs):
    try:
        return operation(*args, **kwargs)
    except (groq.APIError, openai.APIError) as exc:
        status = getattr(exc, "status_code", None)
        # Do not log exception messages, request bodies, keys, or uploaded text.
        LOGGER.warning("Provider failure provider=%s type=%s status=%s",
                       provider, type(exc).__name__, status)
        if status == 404:
            message = ("The configured model or API resource was not found. For Groq, check "
                       "GROQ_MODEL and run Check Groq connection. A model may be unavailable "
                       "to this account; reconnecting the key alone may not fix it.")
        elif status == 401:
            message = "The API key was rejected. The app owner should check the provider key in Streamlit secrets."
        elif status == 403:
            message = "Access was denied. The app owner should check project and model permissions."
        elif status == 429:
            message = "The account is rate-limited or out of quota. Retry later; the app owner should check usage and billing limits."
        elif status in (400, 413, 422):
            message = "The provider rejected the request. Try a shorter question or smaller documents; the owner should check model compatibility."
        elif status and status >= 500:
            message = "The provider is temporarily unavailable. Please try again later."
        else:
            message = "The provider could not complete the request (connection or timeout). Please try again."
        raise ServiceError(f"{provider}: {message}") from None


def create_chat_model(settings: Settings):
    return ChatGroq(api_key=settings.require("Groq"), model=settings.model,
                    base_url="https://api.groq.com", temperature=0,
                    max_tokens=2048, timeout=60, max_retries=1,
                    reasoning_effort="low" if settings.model.startswith("openai/gpt-oss-") else None)


def create_embeddings(settings: Settings):
    return OpenAIEmbeddings(api_key=settings.require("OpenAI"),
                            base_url="https://api.openai.com/v1",
                            model="text-embedding-3-small", dimensions=512, request_timeout=60,
                            max_retries=1, chunk_size=64)


def check_groq(settings: Settings):
    """Check authentication, model visibility, and generation with synthetic text only."""
    with groq.Groq(api_key=settings.require("Groq"),
                   base_url="https://api.groq.com", timeout=20, max_retries=0) as client:
        models = provider_call("Groq", client.models.list)
        available = {model.id for model in models.data}
        if settings.model not in available:
            raise ServiceError("Groq: The configured GROQ_MODEL is not listed for this account. "
                               "Choose a current text model from the Groq console and update app secrets.")
        provider_call("Groq", client.chat.completions.create, model=settings.model,
                      messages=[{"role": "user", "content": "Reply with OK."}],
                      max_completion_tokens=256)
    return "Groq authentication, model lookup, and generation succeeded."
