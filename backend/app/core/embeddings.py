from functools import lru_cache
from typing import Optional
from sentence_transformers import SentenceTransformer

from ..config import get_settings


class EmbeddingService:
    def __init__(self, model_name: str):
        self._model: Optional[SentenceTransformer] = None
        self._model_name = model_name

    def _load_model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def _truncate_text(self, text: str, max_tokens: int = 512) -> str:
        if not text:
            return ""
        # Simple token approximation: split by spaces and take first max_tokens
        tokens = text.split()
        if len(tokens) <= max_tokens:
            return text
        return " ".join(tokens[:max_tokens])

    def embed_single(self, text: str) -> list[float]:
        if not text or not text.strip():
            return [0.0] * 384  # Default dimension for all-MiniLM-L6-v2
        
        truncated_text = self._truncate_text(text)
        model = self._load_model()
        embedding = model.encode(truncated_text, convert_to_numpy=True)
        return embedding.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        
        # Handle empty strings in batch
        processed_texts = []
        for text in texts:
            if not text or not text.strip():
                processed_texts.append("")  # Will be handled by model
            else:
                processed_texts.append(self._truncate_text(text))
        
        model = self._load_model()
        embeddings = model.encode(processed_texts, convert_to_numpy=True)
        
        # Convert to list of lists
        return [emb.tolist() for emb in embeddings]


@lru_cache
def get_embedding_service() -> EmbeddingService:
    settings = get_settings()
    return EmbeddingService(model_name=settings.EMBEDDING_MODEL)
