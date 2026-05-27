from typing import Optional

from ..config import get_settings
from ..core.embeddings import get_embedding_service
from ..core.vector_store import get_vector_store_client
from ..core.llm import get_groq_client
from ..models.property import Property, PropertySearchRequest, PropertySearchResponse
from ..prompts.property_search import PROPERTY_SEARCH_SYSTEM_PROMPT


class PropertyAgent:
    def __init__(self):
        self.embedding_service = get_embedding_service()
        self.vector_store = get_vector_store_client()
        self.groq_client = get_groq_client()
        self.settings = get_settings()

    def search(self, request: PropertySearchRequest) -> PropertySearchResponse:
        """Search for properties based on user query and filters."""
        try:
            # Step 1: Embed the query
            query_embedding = self.embedding_service.embed_single(request.query)
            
            # Step 2: Build filters for Qdrant
            filters = {}
            if request.city:
                filters["city"] = request.city.value
            if request.bhk:
                filters["bhk"] = request.bhk.value
            if request.min_price_lakhs is not None:
                filters["min_price_lakhs"] = request.min_price_lakhs
            if request.max_price_lakhs is not None:
                filters["max_price_lakhs"] = request.max_price_lakhs
            if request.property_type:
                filters["property_type"] = request.property_type.value
            
            # Step 3: Search Qdrant
            search_results = self.vector_store.search(
                collection=self.settings.QDRANT_PROPERTIES_COLLECTION,
                query_vector=query_embedding,
                filters=filters if filters else None,
                top_k=request.top_k,
            )
            
            # Step 4: Deserialize hits back to Property objects
            properties = []
            for hit in search_results:
                try:
                    property_obj = Property(**hit["payload"])
                    properties.append(property_obj)
                except Exception as e:
                    print(f"Failed to deserialize property: {e}")
                    continue
            
            # Step 5: Handle no results
            if not properties:
                return PropertySearchResponse(
                    results=[],
                    llm_summary="No properties found matching your criteria.",
                    total_found=0,
                )
            
            # Step 6: Build context string
            context_parts = ["Retrieved Properties:"]
            for idx, prop in enumerate(properties, 1):
                context_parts.append(f"[Property {idx}]")
                context_parts.append(f"Title: {prop.title}")
                context_parts.append(f"City: {prop.city.value}")
                context_parts.append(f"BHK: {prop.bhk.value}")
                context_parts.append(f"Price: ₹{prop.price_lakhs} lakhs")
                context_parts.append(f"Area: {prop.area_sqft} sqft")
                context_parts.append(f"Locality: {prop.locality}")
                context_parts.append(f"Type: {prop.property_type.value}")
                if prop.amenities:
                    context_parts.append(f"Amenities: {', '.join(prop.amenities)}")
                context_parts.append(f"Description: {prop.description}")
                context_parts.append("")  # Empty line between properties
            
            context = "\n".join(context_parts)
            
            # Step 7: Call Groq for LLM summary
            user_message = f"User query: {request.query}\n\n{context}"
            llm_summary = self.groq_client.chat(
                messages=[{"role": "user", "content": user_message}],
                system_prompt=PROPERTY_SEARCH_SYSTEM_PROMPT,
            )
            
            # Step 8: Return response
            return PropertySearchResponse(
                results=properties,
                llm_summary=llm_summary,
                total_found=len(properties),
            )
            
        except Exception as e:
            print(f"Error in property search: {e}")
            return PropertySearchResponse(
                results=[],
                llm_summary=f"An error occurred during search: {str(e)}",
                total_found=0,
            )
