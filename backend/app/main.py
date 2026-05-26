from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    # Startup
    print(f"Starting Real Estate Copilot - Debug: {settings.DEBUG}")
    yield
    # Shutdown
    print("Shutting down Real Estate Copilot")


def create_app() -> FastAPI:
    settings = get_settings()
    
    app = FastAPI(
        title="Real Estate Copilot",
        description="AI-powered real estate assistant for India",
        version="0.1.0",
        lifespan=lifespan,
    )
    
    # CORS setup for Streamlit frontend (localhost:8501)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:8501"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    @app.get("/health")
    async def health():
        return {"status": "ok", "app": "Real Estate Copilot"}
    
    return app


app = create_app()
