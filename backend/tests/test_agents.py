import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from backend.app.agents.property_agent import PropertyAgent
from backend.app.agents.compare_agent import CompareAgent
from backend.app.agents.faq_agent import FAQAgent
from backend.app.models.property import Property, PropertyType, BHK, City, PropertySearchRequest, PropertySearchResponse


@pytest.fixture
def mock_embedding_service():
    with patch("backend.app.agents.property_agent.get_embedding_service") as mock:
        service = Mock()
        service.embed_single.return_value = [0.1] * 384
        mock.return_value = service
        yield service


@pytest.fixture
def mock_vector_store():
    with patch("backend.app.agents.property_agent.get_vector_store_client") as mock:
        client = Mock()
        mock.return_value = client
        yield client


@pytest.fixture
def mock_groq_client():
    with patch("backend.app.agents.property_agent.get_groq_client") as mock:
        client = Mock()
        client.chat.return_value = "Test LLM summary"
        mock.return_value = client
        yield client


@pytest.fixture
def sample_property():
    return Property(
        id="property_1",
        title="Test Property",
        city=City.BANGALORE,
        locality="Whitefield",
        bhk=BHK.TWO,
        price_lakhs=50.0,
        area_sqft=1200,
        property_type=PropertyType.APARTMENT,
        description="Test description",
    )


class TestPropertyAgent:
    def test_search_calls_vector_store_with_correct_filters(
        self, mock_embedding_service, mock_vector_store, mock_groq_client
    ):
        mock_vector_store.search.return_value = [
            {
                "payload": {
                    "id": "property_1",
                    "title": "Test Property",
                    "city": "Bangalore",
                    "locality": "Whitefield",
                    "bhk": 2,
                    "price_lakhs": 50.0,
                    "area_sqft": 1200,
                    "property_type": "apartment",
                    "amenities": [],
                    "description": "Test description",
                },
                "score": 0.95,
            }
        ]
        
        agent = PropertyAgent()
        request = PropertySearchRequest(
            query="2BHK in Bangalore",
            city=City.BANGALORE,
            bhk=BHK.TWO,
            min_price_lakhs=30,
            max_price_lakhs=80,
            top_k=5,
        )
        
        response = agent.search(request)
        
        mock_vector_store.search.assert_called_once()
        call_args = mock_vector_store.search.call_args
        assert call_args[1]["top_k"] == 5
        filters = call_args[1]["filters"]
        assert filters["city"] == "Bangalore"
        assert filters["bhk"] == 2
        assert filters["min_price_lakhs"] == 30
        assert filters["max_price_lakhs"] == 80

    def test_search_calls_groq_with_correct_system_prompt(
        self, mock_embedding_service, mock_vector_store, mock_groq_client
    ):
        mock_vector_store.search.return_value = [
            {
                "payload": {
                    "id": "property_1",
                    "title": "Test Property",
                    "city": "Bangalore",
                    "locality": "Whitefield",
                    "bhk": 2,
                    "price_lakhs": 50.0,
                    "area_sqft": 1200,
                    "property_type": "apartment",
                    "amenities": [],
                    "description": "Test description",
                },
                "score": 0.95,
            }
        ]
        
        agent = PropertyAgent()
        request = PropertySearchRequest(query="2BHK in Bangalore", top_k=5)
        
        response = agent.search(request)
        
        mock_groq_client.chat.assert_called_once()
        call_args = mock_groq_client.chat.call_args
        assert "system_prompt" in call_args[1]
        assert "Retrieved Properties:" in call_args[1]["messages"][0]["content"]

    def test_search_returns_correct_response_structure(
        self, mock_embedding_service, mock_vector_store, mock_groq_client
    ):
        mock_vector_store.search.return_value = [
            {
                "payload": {
                    "id": "property_1",
                    "title": "Test Property",
                    "city": "Bangalore",
                    "locality": "Whitefield",
                    "bhk": 2,
                    "price_lakhs": 50.0,
                    "area_sqft": 1200,
                    "property_type": "apartment",
                    "amenities": [],
                    "description": "Test description",
                },
                "score": 0.95,
            }
        ]
        
        agent = PropertyAgent()
        request = PropertySearchRequest(query="2BHK in Bangalore", top_k=5)
        
        response = agent.search(request)
        
        assert isinstance(response, PropertySearchResponse)
        assert len(response.results) == 1
        assert response.llm_summary == "Test LLM summary"
        assert response.total_found == 1


class TestFAQAgent:
    def test_answer_calls_qdrant_for_legal_faq_collection(self):
        with patch("backend.app.agents.faq_agent.get_embedding_service") as mock_emb, \
             patch("backend.app.agents.faq_agent.get_vector_store_client") as mock_vs, \
             patch("backend.app.agents.faq_agent.get_groq_client") as mock_groq:
            
            mock_emb.return_value.embed_single.return_value = [0.1] * 384
            mock_vs.return_value.search.return_value = [
                {
                    "payload": {
                        "text": "Test chunk",
                        "source": "rera_guide",
                        "doc_title": "Rera Guide",
                        "chunk_index": 0,
                    },
                    "score": 0.9,
                }
            ]
            mock_groq.return_value.chat.return_value = "Test answer"
            
            agent = FAQAgent()
            response = agent.answer("What is RERA?")
            
            mock_vs.return_value.search.assert_called_once()
            call_args = mock_vs.return_value.search.call_args
            assert "legal_faq" in str(call_args)

    def test_answer_populates_sources_list(self):
        with patch("backend.app.agents.faq_agent.get_embedding_service") as mock_emb, \
             patch("backend.app.agents.faq_agent.get_vector_store_client") as mock_vs, \
             patch("backend.app.agents.faq_agent.get_groq_client") as mock_groq:
            
            mock_emb.return_value.embed_single.return_value = [0.1] * 384
            mock_vs.return_value.search.return_value = [
                {
                    "payload": {
                        "text": "Test chunk 1",
                        "source": "rera_guide",
                        "doc_title": "Rera Guide",
                        "chunk_index": 0,
                    },
                    "score": 0.9,
                },
                {
                    "payload": {
                        "text": "Test chunk 2",
                        "source": "stamp_duty",
                        "doc_title": "Stamp Duty",
                        "chunk_index": 0,
                    },
                    "score": 0.85,
                }
            ]
            mock_groq.return_value.chat.return_value = "Test answer"
            
            agent = FAQAgent()
            response = agent.answer("What is RERA?")
            
            assert len(response["sources"]) == 2
            assert "rera_guide" in response["sources"]
            assert "stamp_duty" in response["sources"]

    def test_answer_deduplicates_same_source_chunks(self):
        with patch("backend.app.agents.faq_agent.get_embedding_service") as mock_emb, \
             patch("backend.app.agents.faq_agent.get_vector_store_client") as mock_vs, \
             patch("backend.app.agents.faq_agent.get_groq_client") as mock_groq:
            
            mock_emb.return_value.embed_single.return_value = [0.1] * 384
            mock_vs.return_value.search.return_value = [
                {
                    "payload": {
                        "text": "Test chunk 1",
                        "source": "rera_guide",
                        "doc_title": "Rera Guide",
                        "chunk_index": 0,
                    },
                    "score": 0.9,
                },
                {
                    "payload": {
                        "text": "Test chunk 2",
                        "source": "rera_guide",
                        "doc_title": "Rera Guide",
                        "chunk_index": 1,
                    },
                    "score": 0.85,
                },
                {
                    "payload": {
                        "text": "Test chunk 3",
                        "source": "rera_guide",
                        "doc_title": "Rera Guide",
                        "chunk_index": 2,
                    },
                    "score": 0.8,
                }
            ]
            mock_groq.return_value.chat.return_value = "Test answer"
            
            agent = FAQAgent()
            response = agent.answer("What is RERA?")
            
            # Should keep at most 2 chunks per source
            assert response["chunks_retrieved"] == 2


class TestCompareAgent:
    def test_compare_raises_value_error_with_fewer_than_2_ids(self):
        with patch("backend.app.agents.compare_agent.get_vector_store_client") as mock_vs, \
             patch("backend.app.agents.compare_agent.get_groq_client") as mock_groq:
            
            agent = CompareAgent()
            
            with pytest.raises(ValueError, match="Need at least 2 properties"):
                agent.compare(["property_1"])
            
            with pytest.raises(ValueError, match="Need at least 2 properties"):
                agent.compare([])

    def test_compare_calculates_price_per_sqft_correctly(self):
        with patch("backend.app.agents.compare_agent.get_vector_store_client") as mock_vs, \
             patch("backend.app.agents.compare_agent.get_groq_client") as mock_groq:
            
            mock_vs.return_value.get_by_id.side_effect = [
                {
                    "payload": {
                        "id": "property_1",
                        "title": "Property A",
                        "city": "Bangalore",
                        "locality": "Whitefield",
                        "bhk": 2,
                        "price_lakhs": 50.0,
                        "area_sqft": 1000,
                        "property_type": "apartment",
                        "amenities": [],
                        "description": "Test",
                        "year_built": 2020,
                    }
                },
                {
                    "payload": {
                        "id": "property_2",
                        "title": "Property B",
                        "city": "Bangalore",
                        "locality": "HSR",
                        "bhk": 2,
                        "price_lakhs": 60.0,
                        "area_sqft": 1200,
                        "property_type": "apartment",
                        "amenities": [],
                        "description": "Test",
                        "year_built": 2018,
                    }
                }
            ]
            mock_groq.return_value.chat.return_value = "Comparison result"
            
            agent = CompareAgent()
            result = agent.compare(["property_1", "property_2"])
            
            # 50 lakhs * 100000 / 1000 = 5000
            assert result["metrics_a"]["price_per_sqft"] == 5000.0
            # 60 lakhs * 100000 / 1200 = 5000
            assert result["metrics_b"]["price_per_sqft"] == 5000.0

    def test_compare_return_dict_has_correct_keys(self):
        with patch("backend.app.agents.compare_agent.get_vector_store_client") as mock_vs, \
             patch("backend.app.agents.compare_agent.get_groq_client") as mock_groq:
            
            mock_vs.return_value.get_by_id.side_effect = [
                {
                    "payload": {
                        "id": "property_1",
                        "title": "Property A",
                        "city": "Bangalore",
                        "locality": "Whitefield",
                        "bhk": 2,
                        "price_lakhs": 50.0,
                        "area_sqft": 1000,
                        "property_type": "apartment",
                        "amenities": [],
                        "description": "Test",
                        "year_built": 2020,
                    }
                },
                {
                    "payload": {
                        "id": "property_2",
                        "title": "Property B",
                        "city": "Bangalore",
                        "locality": "HSR",
                        "bhk": 2,
                        "price_lakhs": 60.0,
                        "area_sqft": 1200,
                        "property_type": "apartment",
                        "amenities": [],
                        "description": "Test",
                        "year_built": 2018,
                    }
                }
            ]
            mock_groq.return_value.chat.return_value = "Comparison result"
            
            agent = CompareAgent()
            result = agent.compare(["property_1", "property_2"])
            
            assert "property_a" in result
            assert "property_b" in result
            assert "metrics_a" in result
            assert "metrics_b" in result
            assert "comparison" in result
            assert isinstance(result["property_a"], Property)
            assert isinstance(result["property_b"], Property)
