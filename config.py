"""Read deployment settings without logging or copying secrets into global state."""
import os
from dataclasses import dataclass, field


class ConfigurationError(Exception):
    pass


def setting(name: str, default: str = "") -> str:
    value = os.environ.get(name)
    if value is None:
        try:
            import streamlit as st
            value = st.secrets.get(name, default)
        except (FileNotFoundError, KeyError):
            value = default
    return str(value or default).strip()


@dataclass(frozen=True)
class Settings:
    groq_key: str = field(repr=False)
    openai_key: str = field(repr=False)
    model: str = "openai/gpt-oss-20b"

    @classmethod
    def load(cls):
        return cls(setting("GROQ_API_KEY"), setting("OPENAI_API_KEY"),
                   setting("GROQ_MODEL", "openai/gpt-oss-20b"))

    def require(self, provider: str):
        key = self.groq_key if provider == "Groq" else self.openai_key
        if not key:
            raise ConfigurationError(f"Configure {provider.upper()}_API_KEY in Streamlit app secrets.")
        return key
