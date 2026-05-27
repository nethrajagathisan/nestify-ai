from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ...agents.faq_agent import FAQAgent


router = APIRouter()


class FAQRequest(BaseModel):
    question: str


class FAQResponse(BaseModel):
    answer: str
    sources: list[str]
    chunks_retrieved: int


@router.post("/faq", response_model=FAQResponse)
async def answer_faq(request: FAQRequest) -> FAQResponse:
    """Answer legal FAQ questions using retrieved context."""
    try:
        agent = FAQAgent()
        result = agent.answer(request.question)
        return FAQResponse(
            answer=result["answer"],
            sources=result["sources"],
            chunks_retrieved=result["chunks_retrieved"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
