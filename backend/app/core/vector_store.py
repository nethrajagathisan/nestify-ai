from functools import lru_cache
from typing import Optional, Any
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    Range,
    MatchValue,
)

from ..config import get_settings


class VectorStoreClient:
    def __init__(self, url: str, api_key: str):
        self._client: Optional[QdrantClient] = None
        self._url = url
        self._api_key = api_key

    def _get_client(self) -> QdrantClient:
        if self._client is None:
            self._client = QdrantClient(
                url=self._url,
                api_key=self._api_key,
                timeout=30,
            )
        return self._client

    def create_collection(self, name: str, vector_size: int) -> None:
        client = self._get_client()
        try:
            collections = client.get_collections().collections
            collection_names = [col.name for col in collections]
            
            if name not in collection_names:
                client.create_collection(
                    collection_name=name,
                    vectors_config=VectorParams(
                        size=vector_size,
                        distance=Distance.COSINE,
                    ),
                )
        except Exception as e:
            raise RuntimeError(f"Failed to create collection {name}: {str(e)}")

    def upsert_points(self, collection: str, points: list[dict]) -> None:
        client = self._get_client()
        try:
            qdrant_points = [
                PointStruct(
                    id=point["id"],
                    vector=point["vector"],
                    payload=point.get("payload", {}),
                )
                for point in points
            ]
            client.upsert(
                collection_name=collection,
                points=qdrant_points,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to upsert points to {collection}: {str(e)}")

    def _build_filter(self, filters: dict) -> Optional[Filter]:
        if not filters:
            return None

        conditions = []

        # City filter (exact match)
        if "city" in filters and filters["city"]:
            conditions.append(
                FieldCondition(
                    key="city",
                    match=MatchValue(value=filters["city"]),
                )
            )

        # BHK filter (exact match)
        if "bhk" in filters and filters["bhk"] is not None:
            conditions.append(
                FieldCondition(
                    key="bhk",
                    match=MatchValue(value=filters["bhk"]),
                )
            )

        # Property type filter (exact match)
        if "property_type" in filters and filters["property_type"]:
            conditions.append(
                FieldCondition(
                    key="property_type",
                    match=MatchValue(value=filters["property_type"]),
                )
            )

        # Price range filters
        range_conditions = {}
        if "min_price_lakhs" in filters and filters["min_price_lakhs"] is not None:
            range_conditions["gte"] = filters["min_price_lakhs"]
        if "max_price_lakhs" in filters and filters["max_price_lakhs"] is not None:
            range_conditions["lte"] = filters["max_price_lakhs"]
        
        if range_conditions:
            conditions.append(
                FieldCondition(
                    key="price_lakhs",
                    range=Range(**range_conditions),
                )
            )

        if conditions:
            return Filter(must=conditions)
        return None

    def search(
        self,
        collection: str,
        query_vector: list[float],
        filters: Optional[dict],
        top_k: int,
    ) -> list[dict]:
        client = self._get_client()
        try:
            qdrant_filter = self._build_filter(filters)
            
            results = client.search(
                collection_name=collection,
                query_vector=query_vector,
                query_filter=qdrant_filter,
                limit=top_k,
                with_payload=True,
                with_score=True,
            )

            return [
                {
                    "payload": hit.payload,
                    "score": hit.score,
                    "id": hit.id,
                }
                for hit in results
            ]
        except Exception as e:
            raise RuntimeError(f"Failed to search in {collection}: {str(e)}")

    def get_by_id(self, collection: str, point_id: str | int) -> Optional[dict]:
        client = self._get_client()
        try:
            result = client.retrieve(
                collection_name=collection,
                ids=[point_id],
                with_payload=True,
            )
            
            if result:
                return {
                    "payload": result[0].payload,
                    "id": result[0].id,
                }
            return None
        except Exception as e:
            raise RuntimeError(f"Failed to get point {point_id} from {collection}: {str(e)}")


@lru_cache
def get_vector_store_client() -> VectorStoreClient:
    settings = get_settings()
    return VectorStoreClient(
        url=settings.QDRANT_URL,
        api_key=settings.QDRANT_API_KEY,
    )
