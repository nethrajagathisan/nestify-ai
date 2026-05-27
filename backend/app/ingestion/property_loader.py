import json
import uuid
from pathlib import Path

from ..config import get_settings
from ..core.embeddings import get_embedding_service
from ..core.vector_store import get_vector_store_client


def _make_embedding_text(prop: dict) -> str:
    """Build a rich text string for embedding from property fields."""
    parts = [
        prop.get("title", ""),
        f"City: {prop.get('city', '')}",
        f"Locality: {prop.get('locality', '')}",
        f"BHK: {prop.get('bhk', '')}",
        f"Price: {prop.get('price_lakhs', '')} lakhs",
        f"Area: {prop.get('area_sqft', '')} sqft",
        f"Type: {prop.get('property_type', '')}",
        f"Amenities: {', '.join(prop.get('amenities', []))}",
        prop.get("description", ""),
    ]
    return " | ".join(p for p in parts if p)


def load_properties() -> None:
    """Load properties from listings.json, embed, and upsert to Qdrant."""
    settings = get_settings()
    embedding_service = get_embedding_service()
    vector_store = get_vector_store_client()

    listings_path = Path("data/properties/listings.json")
    if not listings_path.exists():
        raise FileNotFoundError(f"listings.json not found at {listings_path}")

    with open(listings_path, "r", encoding="utf-8") as f:
        listings = json.load(f)

    if not listings:
        print("No listings found in listings.json")
        return

    print(f"Loaded {len(listings)} properties from {listings_path}")

    # Build embedding texts
    texts = [_make_embedding_text(p) for p in listings]

    print(f"Embedding {len(texts)} properties...")
    embeddings = embedding_service.embed_batch(texts)

    vector_size = len(embeddings[0]) if embeddings else 384
    print(f"Creating/recreating collection '{settings.QDRANT_PROPERTIES_COLLECTION}'...")
    vector_store.create_collection(
        name=settings.QDRANT_PROPERTIES_COLLECTION,
        vector_size=vector_size,
    )

    # Build points — use UUID as Qdrant ID and store it as the payload id too
    points = []
    for prop, embedding in zip(listings, embeddings):
        point_id = str(uuid.uuid4())
        payload = dict(prop)
        payload["id"] = point_id       # overwrite prop_XXX with UUID so compare works
        points.append({
            "id": point_id,
            "vector": embedding,
            "payload": payload,
        })

    print(f"Upserting {len(points)} points to Qdrant...")
    # Upsert in batches of 100 to avoid timeouts
    batch_size = 100
    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        vector_store.upsert_points(
            collection=settings.QDRANT_PROPERTIES_COLLECTION,
            points=batch,
        )
        print(f"  Upserted {min(i + batch_size, len(points))}/{len(points)}")

    print(f"Done. {len(points)} properties ingested into '{settings.QDRANT_PROPERTIES_COLLECTION}'.")
