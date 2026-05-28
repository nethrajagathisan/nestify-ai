import logging
from typing import Optional

from ..config import get_settings
from ..core.embeddings import get_embedding_service
from ..core.vector_store import get_vector_store_client
from ..core.llm import get_groq_client
from ..core.bm25_store import get_bm25_store
from ..core.rrf_merger import hybrid_search
from ..core.reranker import get_reranker
from ..core.query_expansion import QueryExpander
from ..models.property import Property, PropertySearchRequest, PropertySearchResponse
from ..prompts.property_search import PROPERTY_SEARCH_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class PropertyAgent:
    def __init__(self):
        self.embedding_service = get_embedding_service()
        self.vector_store = get_vector_store_client()
        self.groq_client = get_groq_client()
        self.bm25_store = get_bm25_store()
        self.reranker = get_reranker()
        self.settings = get_settings()

    def search(
        self,
        request: PropertySearchRequest,
        use_query_expansion: bool = True,
        use_reranking: bool = True,
        top_k_final: int = 5
    ) -> PropertySearchResponse:
        """
        Search for properties using advanced RAG pipeline.
        
        Pipeline:
        1. Query Expansion (optional): Generate semantic variations
        2. Hybrid Search: Dense (Qdrant) + Sparse (BM25) with RRF merging
        3. Reranking (optional): Cross-encoder reranking for top results
        4. Context Assembly: Augment with parent context if available
        5. LLM Summary: Generate natural language response
        
        Benefits:
        - Query expansion handles paraphrasing ("2 BHK" vs "two-bedroom")
        - Hybrid search catches both semantic and keyword matches
        - Reranking ensures top results are truly relevant
        - Parent-child chunking provides better context
        
        Args:
            request: Property search request with query and filters
            use_query_expansion: Enable query expansion (default True)
            use_reranking: Enable cross-encoder reranking (default True)
            top_k_final: Number of final results to return (default 5)
        
        Returns:
            PropertySearchResponse with results and LLM summary
        """
        try:
            logger.debug(f"Starting property search with query: {request.query}")
            
            # Build filters for Qdrant
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
            
            logger.debug(f"Built filters: {filters}")
            
            # Step 1: Embed the query
            query_embedding = self.embedding_service.embed_single(request.query)
            logger.debug("Query embedded successfully")
            
            # Step 2: Query Expansion (optional)
            if use_query_expansion:
                logger.debug("Step 1: Query expansion enabled")
                expander = QueryExpander(self.groq_client)
                query_variations = expander.expand_query(request.query, num_variations=3)
                logger.debug(f"Generated {len(query_variations)} query variations")
            else:
                logger.debug("Step 1: Query expansion disabled, using original query")
                query_variations = [request.query]
            
            # Step 3: Hybrid Search with RRF
            logger.debug("Step 2: Hybrid search (dense + sparse with RRF)")
            
            if use_query_expansion:
                # Multi-query retrieval: search each variation and merge
                from ..core.query_expansion import multi_query_retrieval
                
                search_results = multi_query_retrieval(
                    query=request.query,
                    vector_store=self.vector_store,
                    bm25_store=self.bm25_store,
                    reranker=self.reranker if use_reranking else None,
                    llm_client=self.groq_client,
                    query_vector=query_embedding,
                    collection=self.settings.QDRANT_PROPERTIES_COLLECTION,
                    num_variations=3,
                    top_k_per_query=20,
                    top_k_final=top_k_final,
                    filters=filters if filters else None
                )
                logger.debug(f"Multi-query retrieval returned {len(search_results)} results")
            else:
                # Single query hybrid search
                search_results = hybrid_search(
                    query=request.query,
                    vector_store=self.vector_store,
                    bm25_store=self.bm25_store,
                    query_vector=query_embedding,
                    collection=self.settings.QDRANT_PROPERTIES_COLLECTION,
                    top_k_dense=20,
                    top_k_sparse=20,
                    top_k_final=20 if use_reranking else top_k_final,
                    filters=filters if filters else None
                )
                logger.debug(f"Hybrid search returned {len(search_results)} results")
            
            # Step 4: Reranking (optional)
            if use_reranking and not use_query_expansion:
                logger.debug("Step 3: Reranking with cross-encoder")
                search_results = self.reranker.rerank_with_metadata(
                    query=request.query,
                    results=search_results,
                    top_k=top_k_final
                )
                logger.debug(f"Reranking complete, top {len(search_results)} results")
            
            # Step 5: Context Assembly with parent context
            logger.debug("Step 4: Context assembly")
            properties = []
            for result in search_results:
                try:
                    # Check if parent context is available
                    text = result.get("text", "")
                    parent = result.get("parent", "")
                    
                    # Use parent context if available, otherwise use chunk text
                    context_text = parent if parent else text
                    
                    # Deserialize to Property object
                    # Note: This assumes the payload structure matches Property model
                    # If using parent-child chunks, we may need to adjust this
                    metadata = result.get("metadata", {})
                    
                    # Try to create Property from metadata or result
                    if "title" in metadata:
                        property_obj = Property(**metadata)
                    else:
                        # Fallback: create minimal property from result
                        property_obj = Property(
                            title=result.get("id", "Unknown"),
                            description=context_text,
                            city=request.city or "Unknown",
                            bhk=request.bhk or 1,
                            price_lakhs=0,
                            area_sqft=0,
                            locality="Unknown",
                            property_type=request.property_type or "apartment"
                        )
                    
                    properties.append(property_obj)
                except Exception as e:
                    logger.warning(f"Failed to deserialize property: {e}")
                    continue
            
            logger.debug(f"Deserialized {len(properties)} property objects")
            
            # Handle no results
            if not properties:
                logger.warning("No properties found")
                return PropertySearchResponse(
                    results=[],
                    llm_summary="No properties found matching your criteria.",
                    total_found=0,
                )
            
            # Step 6: Build context string for LLM
            logger.debug("Step 5: Building context for LLM summary")
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
            logger.debug(f"Context built with {len(context_parts)} parts")
            
            # Step 7: Call Groq for LLM summary
            logger.debug("Step 6: Generating LLM summary")
            user_message = f"User query: {request.query}\n\n{context}"
            llm_summary = self.groq_client.chat(
                messages=[{"role": "user", "content": user_message}],
                system_prompt=PROPERTY_SEARCH_SYSTEM_PROMPT,
            )
            logger.debug("LLM summary generated")
            
            # Return response
            logger.debug(f"Search complete, returning {len(properties)} results")
            return PropertySearchResponse(
                results=properties,
                llm_summary=llm_summary,
                total_found=len(properties),
            )
            
        except Exception as e:
            logger.error(f"Error in property search: {e}", exc_info=True)
            return PropertySearchResponse(
                results=[],
                llm_summary=f"An error occurred during search: {str(e)}",
                total_found=0,
            )
