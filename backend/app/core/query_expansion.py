"""
Query expansion for improved retrieval recall.

This module provides query expansion capabilities using LLM to generate
semantic variations of user queries, improving recall for paraphrases,
abbreviations, and alternative phrasings.

Benefits:
- Handles query paraphrasing: "2 BHK" vs "two-bedroom"
- Handles abbreviations: "BHK" vs "bedroom hall kitchen"
- Improves recall on semantic queries

Example Flow:
```
User: "What are cheap apartments in Bangalore?"
Expanded to:
  1. "What are cheap apartments in Bangalore?"
  2. "Find affordable apartments in Bangalore"
  3. "Budget-friendly flats in Bangalore"
  4. "Low-cost residential properties Bangalore"

Hybrid search each → merge → deduplicate → rerank → return top-5
```
"""

import json
import logging
from typing import Dict, List, Optional

from .llm import GroqClient
from .rrf_merger import hybrid_search
from .reranker import RerankerService
from .vector_store import VectorStoreClient
from .bm25_store import BM25Store

logger = logging.getLogger(__name__)


class QueryExpander:
    """
    Query expansion service using LLM to generate semantic variations.
    
    Generates alternative phrasings of queries to improve retrieval recall
    by capturing paraphrases, abbreviations, and semantic variations.
    """

    def __init__(self, llm_client: GroqClient):
        """
        Initialize QueryExpander with LLM client.
        
        Args:
            llm_client: GroqClient instance for generating query variations
        """
        self._llm_client = llm_client

    def expand_query(
        self,
        query: str,
        num_variations: int = 3
    ) -> List[str]:
        """
        Generate semantic variations of a query using LLM.
        
        Args:
            query: Original user query
            num_variations: Number of variations to generate (default: 3)
        
        Returns:
            List of query variations including original query
            Format: [original_query, variation_1, variation_2, ...]
        
        Raises:
            ValueError: If query is empty
            RuntimeError: If LLM fails to generate valid JSON
        
        Example:
            >>> expander = QueryExpander(llm_client)
            >>> variations = expander.expand_query("2 BHK apartment in Bangalore under 80 lakhs")
            >>> variations
            [
                "2 BHK apartment in Bangalore under 80 lakhs",
                "Find 2-bedroom flats in Bangalore with max price 80 lakhs",
                "2 bedroom residential property Bangalore budget 80 lakhs",
                "Bangalore 2 BHK flat under 80 lakh price"
            ]
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        system_prompt = f"""You are a query expansion expert. Given a user query, generate {num_variations} alternative phrasings that would retrieve the same information.

Return ONLY the variations as a JSON list of strings, no explanations.
Example:
User query: "2 BHK apartment in Bangalore under 80 lakhs"
Variations:
[
  "2 BHK apartment in Bangalore under 80 lakhs",
  "Find 2-bedroom flats in Bangalore with max price 80 lakhs",
  "2 bedroom residential property Bangalore budget 80 lakhs",
  "Bangalore 2 BHK flat under 80 lakh price"
]"""
        
        try:
            response = self._llm_client.chat(
                messages=[{"role": "user", "content": query}],
                system_prompt=system_prompt
            )
            
            # Parse JSON response
            variations = json.loads(response.strip())
            
            # Validate response is a list of strings
            if not isinstance(variations, list):
                raise ValueError("LLM response is not a list")
            
            if not all(isinstance(v, str) for v in variations):
                raise ValueError("LLM response contains non-string elements")
            
            # Ensure original query is included
            if query not in variations:
                variations.insert(0, query)
            
            logger.info(f"Generated {len(variations)} query variations for: {query[:50]}...")
            return variations
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            logger.error(f"Raw response: {response}")
            # Fallback: return original query only
            return [query]
        except Exception as e:
            logger.error(f"Query expansion failed: {e}")
            # Fallback: return original query only
            return [query]


def multi_query_retrieval(
    query: str,
    vector_store: VectorStoreClient,
    bm25_store: BM25Store,
    reranker: RerankerService,
    llm_client: GroqClient,
    query_vector: List[float],
    collection: str = "properties",
    num_variations: int = 3,
    top_k_per_query: int = 10,
    top_k_final: int = 5,
    filters: Optional[Dict] = None
) -> List[Dict]:
    """
    Perform multi-query retrieval with query expansion.
    
    Expands the query into semantic variations, performs hybrid search
    for each variation, merges results, deduplicates, and reranks.
    
    Args:
        query: Original user query
        vector_store: VectorStoreClient for dense search
        bm25_store: BM25Store for sparse search
        reranker: RerankerService for final reranking
        llm_client: GroqClient for query expansion
        query_vector: Query embedding vector for dense search
        collection: Qdrant collection name (default "properties")
        num_variations: Number of query variations to generate (default: 3)
        top_k_per_query: Number of results per query variation (default: 10)
        top_k_final: Number of final results to return (default: 5)
        filters: Optional filters for dense search
    
    Returns:
        Top-k reranked results from all query variations
        Format: [{"id", "score", "text", "metadata", ...}]
    
    Example:
        >>> from backend.app.core.llm import get_groq_client
        >>> from backend.app.core.query_expansion import QueryExpander
        >>> 
        >>> llm_client = get_groq_client()
        >>> expander = QueryExpander(llm_client)
        >>> 
        >>> results = multi_query_retrieval(
        ...     query="What are cheap apartments in Bangalore?",
        ...     vector_store=vector_client,
        ...     bm25_store=bm25,
        ...     reranker=reranker,
        ...     llm_client=llm_client,
        ...     query_vector=query_embedding,
        ...     num_variations=3,
        ...     top_k_final=5
        ... )
        >>> len(results)
        5
    """
    # Expand query
    expander = QueryExpander(llm_client)
    query_variations = expander.expand_query(query, num_variations=num_variations)
    
    logger.info(f"Query expanded to {len(query_variations)} variations")
    
    # Collect results from all query variations
    all_results: Dict[str, Dict] = {}
    
    for variation in query_variations:
        try:
            # Perform hybrid search for this variation
            results = hybrid_search(
                query=variation,
                vector_store=vector_store,
                bm25_store=bm25_store,
                query_vector=query_vector,
                collection=collection,
                top_k_dense=top_k_per_query,
                top_k_sparse=top_k_per_query,
                top_k_final=top_k_per_query,
                filters=filters
            )
            
            # Merge results, keeping highest score for each document
            for result in results:
                doc_id = result.get("id")
                if doc_id is None:
                    continue
                
                if doc_id not in all_results:
                    all_results[doc_id] = result
                else:
                    # Keep the result with higher RRF score
                    if result.get("rrf_score", 0) > all_results[doc_id].get("rrf_score", 0):
                        all_results[doc_id] = result
                        
        except Exception as e:
            logger.error(f"Hybrid search failed for variation '{variation}': {e}")
            continue
    
    if not all_results:
        logger.warning("No results collected from any query variation")
        return []
    
    logger.info(f"Collected {len(all_results)} unique results after deduplication")
    
    # Convert to list and rerank
    merged_results = list(all_results.values())
    
    try:
        # Rerank merged results using original query
        reranked = reranker.rerank_with_metadata(
            query=query,
            results=merged_results,
            top_k=top_k_final
        )
        
        logger.info(f"Reranking complete. Returning top {len(reranked)} results")
        return reranked
        
    except Exception as e:
        logger.error(f"Reranking failed: {e}")
        # Fallback: return results sorted by RRF score
        merged_results.sort(key=lambda x: x.get("rrf_score", 0), reverse=True)
        return merged_results[:top_k_final]
