from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ...agents.property_agent import PropertyAgent
from ...agents.compare_agent import CompareAgent
from ...models.property import PropertySearchRequest, PropertySearchResponse


router = APIRouter()


class CompareRequest(BaseModel):
    property_ids: list[str]


@router.post("/search", response_model=PropertySearchResponse)
async def search_properties(request: PropertySearchRequest) -> PropertySearchResponse:
    """Search for properties based on query and filters."""
    try:
        agent = PropertyAgent()
        return agent.search(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compare")
async def compare_properties(request: CompareRequest) -> dict:
    """Compare multiple properties by ID."""
    try:
        agent = CompareAgent()
        return agent.compare(request.property_ids)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
