from datetime import datetime
from typing import Optional

from ..config import get_settings
from ..core.vector_store import get_vector_store_client
from ..core.llm import get_groq_client
from ..models.property import Property
from ..prompts.comparison import COMPARISON_SYSTEM_PROMPT


class CompareAgent:
    def __init__(self):
        self.vector_store = get_vector_store_client()
        self.groq_client = get_groq_client()
        self.settings = get_settings()

    def compare(self, property_ids: list[str]) -> dict:
        """Compare multiple properties by ID."""
        # Step 1: Validate input
        if len(property_ids) < 2:
            raise ValueError("Need at least 2 properties to compare")
        
        try:
            # Step 2: Fetch each property by ID from Qdrant
            properties = []
            for prop_id in property_ids:
                result = self.vector_store.get_by_id(
                    collection=self.settings.QDRANT_PROPERTIES_COLLECTION,
                    point_id=prop_id,
                )
                if result:
                    property_obj = Property(**result["payload"])
                    properties.append(property_obj)
                else:
                    raise ValueError(f"Property with ID {prop_id} not found")
            
            if len(properties) < 2:
                raise ValueError("Could not retrieve enough properties for comparison")
            
            # Step 3: Calculate metrics for each property
            current_year = datetime.now().year
            metrics_list = []
            
            for prop in properties:
                price_per_sqft = (prop.price_lakhs * 100000) / prop.area_sqft
                age_years = current_year - prop.year_built if prop.year_built else None
                
                metrics_list.append({
                    "price_per_sqft": round(price_per_sqft, 2),
                    "age_years": age_years,
                })
            
            # Step 4: Build comparison context
            context_parts = []
            for idx, (prop, metrics) in enumerate(zip(properties, metrics_list), start=65):
                letter = chr(idx)  # A, B, C, etc.
                context_parts.append(f"Property {letter}:")
                context_parts.append(f"Title: {prop.title}")
                context_parts.append(f"City: {prop.city.value}")
                context_parts.append(f"Locality: {prop.locality}")
                context_parts.append(f"BHK: {prop.bhk.value}")
                context_parts.append(f"Price: ₹{prop.price_lakhs}L (₹{metrics['price_per_sqft']}/sqft)")
                context_parts.append(f"Area: {prop.area_sqft} sqft")
                context_parts.append(f"Type: {prop.property_type.value}")
                if metrics["age_years"] is not None:
                    context_parts.append(f"Age: {metrics['age_years']} years")
                if prop.amenities:
                    context_parts.append(f"Amenities: {', '.join(prop.amenities)}")
                context_parts.append(f"Description: {prop.description}")
                context_parts.append("")  # Empty line between properties
            
            context = "\n".join(context_parts)
            
            # Step 5: Call Groq for comparison
            user_message = f"Compare the following properties:\n\n{context}"
            comparison = self.groq_client.chat(
                messages=[{"role": "user", "content": user_message}],
                system_prompt=COMPARISON_SYSTEM_PROMPT,
            )
            
            # Step 6: Return comparison dict
            return {
                "property_a": properties[0],
                "property_b": properties[1] if len(properties) > 1 else None,
                "metrics_a": metrics_list[0],
                "metrics_b": metrics_list[1] if len(metrics_list) > 1 else None,
                "comparison": comparison,
            }
            
        except Exception as e:
            raise RuntimeError(f"Error during property comparison: {str(e)}")
