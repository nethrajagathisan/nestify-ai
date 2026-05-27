"""Tests for FastAPI endpoints."""

from fastapi.testclient import TestClient

from backend.app.main import create_app


class TestHealthEndpoint:
    def setup_method(self):
        app = create_app()
        self.client = TestClient(app)

    def test_health(self):
        response = self.client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["app"] == "Real Estate Copilot"

    def test_cors_headers(self):
        response = self.client.get(
            "/health",
            headers={"Origin": "http://localhost:8501"},
        )
        assert response.status_code == 200
        assert "access-control-allow-origin" in response.headers
        assert response.headers["access-control-allow-origin"] == "http://localhost:8501"
