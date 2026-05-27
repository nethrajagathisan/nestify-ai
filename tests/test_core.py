"""Tests for core services (embeddings, vector store, LLM, config)."""

import pytest
from unittest.mock import patch, MagicMock

from backend.app.config import Settings, get_settings
from backend.app.core.embeddings import EmbeddingService, get_embedding_service
from backend.app.core.vector_store import VectorStoreClient, get_vector_store_client


class TestConfig:
    @patch.dict(
        "os.environ",
        {
            "GROQ_API_KEY": "test_key",
            "QDRANT_URL": "https://test.qdrant.io",
            "QDRANT_API_KEY": "test_qdrant_key",
        },
        clear=True,
    )
    def test_settings_from_env(self):
        settings = Settings()
        assert settings.GROQ_API_KEY == "test_key"
        assert settings.QDRANT_URL == "https://test.qdrant.io"
        assert settings.QDRANT_PROPERTIES_COLLECTION == "properties"
        assert settings.QDRANT_FAQ_COLLECTION == "legal_faq"
        assert settings.EMBEDDING_MODEL == "all-MiniLM-L6-v2"
        assert settings.DEBUG is False


class TestEmbeddingService:
    def test_truncate_text_short(self):
        svc = EmbeddingService(model_name="all-MiniLM-L6-v2")
        text = "Short text"
        result = svc._truncate_text(text, max_tokens=512)
        assert result == "Short text"

    def test_truncate_text_long(self):
        svc = EmbeddingService(model_name="all-MiniLM-L6-v2")
        text = "word " * 600
        result = svc._truncate_text(text, max_tokens=512)
        assert len(result.split()) == 512

    def test_embed_single_empty(self):
        svc = EmbeddingService(model_name="all-MiniLM-L6-v2")
        result = svc.embed_single("")
        assert isinstance(result, list)
        assert len(result) == 384
        assert all(v == 0.0 for v in result)

    def test_embed_batch_empty(self):
        svc = EmbeddingService(model_name="all-MiniLM-L6-v2")
        result = svc.embed_batch([])
        assert result == []

    @patch("backend.app.core.embeddings.SentenceTransformer")
    def test_embed_single_mock(self, mock_model_cls):
        import numpy as np
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([0.1] * 384)
        mock_model_cls.return_value = mock_model

        svc = EmbeddingService(model_name="all-MiniLM-L6-v2")
        result = svc.embed_single("test sentence")
        assert len(result) == 384
        mock_model.encode.assert_called_once()


class TestVectorStoreClient:
    @patch("backend.app.core.vector_store.QdrantClient")
    def test_create_collection_new(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.get_collections.return_value = MagicMock(collections=[])
        mock_client_cls.return_value = mock_client

        store = VectorStoreClient(url="https://test", api_key="key")
        store.create_collection("test_collection", vector_size=384)

        mock_client.create_collection.assert_called_once()

    @patch("backend.app.core.vector_store.QdrantClient")
    def test_create_collection_exists(self, mock_client_cls):
        mock_collection = MagicMock()
        mock_collection.name = "test_collection"
        mock_client = MagicMock()
        mock_client.get_collections.return_value = MagicMock(collections=[mock_collection])
        mock_client_cls.return_value = mock_client

        store = VectorStoreClient(url="https://test", api_key="key")
        store.create_collection("test_collection", vector_size=384)

        mock_client.create_collection.assert_not_called()

    @patch("backend.app.core.vector_store.QdrantClient")
    def test_search(self, mock_client_cls):
        mock_hit = MagicMock()
        mock_hit.payload = {"text": "test", "city": "Bangalore"}
        mock_hit.score = 0.95
        mock_hit.id = "point-1"

        mock_client = MagicMock()
        mock_client.search.return_value = [mock_hit]
        mock_client_cls.return_value = mock_client

        store = VectorStoreClient(url="https://test", api_key="key")
        results = store.search(
            collection="test",
            query_vector=[0.1] * 384,
            filters={"city": "Bangalore"},
            top_k=5,
        )

        assert len(results) == 1
        assert results[0]["payload"]["city"] == "Bangalore"
        assert results[0]["score"] == 0.95

    @patch("backend.app.core.vector_store.QdrantClient")
    def test_build_filter_price_range(self, mock_client_cls):
        store = VectorStoreClient(url="https://test", api_key="key")
        filters = {
            "city": "Bangalore",
            "min_price_lakhs": 50.0,
            "max_price_lakhs": 200.0,
        }
        qfilter = store._build_filter(filters)
        assert qfilter is not None


class TestPrompts:
    def test_property_search_prompt_import(self):
        from backend.app.prompts.property_search import PROPERTY_SEARCH_SYSTEM_PROMPT
        assert "price-per-sqft" in PROPERTY_SEARCH_SYSTEM_PROMPT
        assert "Bangalore" in PROPERTY_SEARCH_SYSTEM_PROMPT
        assert "lakhs" in PROPERTY_SEARCH_SYSTEM_PROMPT

    def test_legal_faq_prompt_import(self):
        from backend.app.prompts.legal_faq import LEGAL_FAQ_SYSTEM_PROMPT
        assert "general information, not legal advice" in LEGAL_FAQ_SYSTEM_PROMPT
        assert "RERA 2016" in LEGAL_FAQ_SYSTEM_PROMPT

    def test_comparison_prompt_import(self):
        from backend.app.prompts.comparison import COMPARISON_SYSTEM_PROMPT
        assert "Price-per-sqft" in COMPARISON_SYSTEM_PROMPT
        assert "suits" in COMPARISON_SYSTEM_PROMPT
