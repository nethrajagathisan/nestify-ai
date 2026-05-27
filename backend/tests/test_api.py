import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch

from backend.app.main import create_app


@pytest.fixture
def client():
    app = create_app()
    return TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_200_and_correct_response(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["app"] == "Real Estate Copilot"


class TestFAQEndpoint:
    def test_faq_endpoint_with_mocked_agent_returns_200(self, client):
        with patch("backend.app.api.routes.faq.FAQAgent") as mock_agent_class:
            mock_agent = Mock()
            mock_agent.answer.return_value = {
                "question": "What is RERA?",
                "answer": "Test answer",
                "sources": ["rera_guide"],
                "chunks_retrieved": 2,
            }
            mock_agent_class.return_value = mock_agent
            
            response = client.post(
                "/api/v1/faq",
                json={"question": "What is RERA?"}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["answer"] == "Test answer"
            assert data["sources"] == ["rera_guide"]
            assert data["chunks_retrieved"] == 2
            mock_agent.answer.assert_called_once_with("What is RERA?")


class TestPropertySearchEndpoint:
    def test_search_endpoint_with_mocked_agent_returns_200(self, client):
        with patch("backend.app.api.routes.properties.PropertyAgent") as mock_agent_class:
            mock_agent = Mock()
            mock_agent.search.return_value = Mock(
                llm_summary="Test summary",
                results=[],
                total_found=0
            )
            mock_agent_class.return_value = mock_agent
            
            response = client.post(
                "/api/v1/properties/search",
                json={
                    "query": "2BHK in Bangalore",
                    "top_k": 5
                }
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["llm_summary"] == "Test summary"
            assert data["total_found"] == 0
            mock_agent.search.assert_called_once()

    def test_search_endpoint_with_invalid_request_returns_422(self, client):
        response = client.post(
            "/api/v1/properties/search",
            json={
                "query": "Test",
                "top_k": 25  # Invalid: > 20
            }
        )
        
        assert response.status_code == 422


class TestCompareEndpoint:
    def test_compare_endpoint_with_mocked_agent_returns_200(self, client):
        from backend.app.models.property import Property, PropertyType, BHK, City
        
        with patch("backend.app.api.routes.properties.CompareAgent") as mock_agent_class:
            mock_agent = Mock()
            mock_agent.compare.return_value = {
                "property_a": Property(
                    id="property_1",
                    title="Property A",
                    city=City.BANGALORE,
                    locality="Whitefield",
                    bhk=BHK.TWO,
                    price_lakhs=50.0,
                    area_sqft=1000,
                    property_type=PropertyType.APARTMENT,
                    description="Test",
                    year_built=2020,
                ),
                "property_b": Property(
                    id="property_2",
                    title="Property B",
                    city=City.BANGALORE,
                    locality="HSR",
                    bhk=BHK.TWO,
                    price_lakhs=60.0,
                    area_sqft=1200,
                    property_type=PropertyType.APARTMENT,
                    description="Test",
                    year_built=2018,
                ),
                "metrics_a": {"price_per_sqft": 5000.0, "age_years": 5},
                "metrics_b": {"price_per_sqft": 4500.0, "age_years": 3},
                "comparison": "Property A is better",
            }
            mock_agent_class.return_value = mock_agent
            
            response = client.post(
                "/api/v1/properties/compare",
                json={"property_ids": ["property_1", "property_2"]}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "comparison" in data
            assert data["comparison"] == "Property A is better"
            mock_agent.compare.assert_called_once_with(["property_1", "property_2"])


class TestChatEndpoint:
    def test_chat_endpoint_with_mocked_orchestrator_returns_200(self, client):
        with patch("backend.app.api.routes.chat.CopilotOrchestrator") as mock_orch_class:
            mock_orch = Mock()
            mock_orch.invoke.return_value = {
                "response": "Test response",
                "intent": "property_search",
                "sources": [],
                "error": None,
            }
            mock_orch_class.return_value = mock_orch
            
            response = client.post(
                "/api/v1/chat",
                json={
                    "query": "Find 2BHK in Bangalore",
                    "history": [],
                    "filters": {}
                }
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["response"] == "Test response"
            assert data["intent"] == "property_search"
            mock_orch.invoke.assert_called_once()


@pytest.mark.slow
class TestSlowTests:
    def test_slow_integration_test(self, client):
        """This test is marked as slow and should be skipped with -m 'not slow'"""
        import time
        time.sleep(0.1)
        assert True
