from functools import lru_cache
from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    GROQ_API_KEY: str
    QDRANT_URL: str
    QDRANT_API_KEY: str
    QDRANT_PROPERTIES_COLLECTION: str = "properties"
    QDRANT_FAQ_COLLECTION: str = "legal_faq"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    GROQ_CHAT_MODEL: str = "llama-3.1-70b-versatile"
    GROQ_CLASSIFIER_MODEL: str = "gemma2-9b-it"
    DEBUG: bool = False


@lru_cache
def get_settings() -> Settings:
    return Settings()
