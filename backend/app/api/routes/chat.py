from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ...agents.router import CopilotOrchestrator


router = APIRouter()


class ChatRequest(BaseModel):
    query: str
    history: list[dict] = []
    filters: dict = {}


class ChatResponse(BaseModel):
    response: str
    intent: str
    sources: list[str]
    error: Optional[str] = None


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Chat endpoint that routes queries to appropriate agents."""
    try:
        orchestrator = CopilotOrchestrator()
        result = orchestrator.invoke(
            query=request.query,
            conversation_history=request.history,
            filters=request.filters,
        )
        return ChatResponse(
            response=result["response"],
            intent=result["intent"],
            sources=result["sources"],
            error=result.get("error"),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
