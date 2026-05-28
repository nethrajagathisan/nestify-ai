"""
RerankerService for cross-encoder based result reranking.

This module provides a RerankerService that uses cross-encoder models
to rerank retrieval results for improved relevance.

Performance Notes:
- Model: cross-encoder/ms-marco-MiniLM-L-6-v2 (~130MB)
- Latency: ~50ms for 20 passages on CPU
- Can batch 100+ passages efficiently
- Lazy loading: model loaded on first use

Full Pipeline Example:
```python
from backend.app.core.rrf_merger import hybrid_search
from backend.app.core.reranker import get_reranker

# 1. Hybrid search returns top-20
results = hybrid_search(
    query="2 BHK apartment in Bangalore",
    vector_store=vector_client,
    bm25_store=bm25_store,
    query_vector=query_embedding,
    top_k_final=20
)

# 2. Rerank to top-5
reranker = get_reranker()
final_results = reranker.rerank_with_metadata(query, results)[:5]

# final_results now contains top-5 most relevant results
```
"""

import logging
from functools import lru_cache
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class RerankerService:
    """
    Cross-encoder based reranker for improving retrieval results.
    
    Uses sentence-transformers CrossEncoder to rerank passages based on
    query-passage relevance scores.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize RerankerService.
        
        Args:
            model_name: Name of the cross-encoder model to use
                       Default: cross-encoder/ms-marco-MiniLM-L-6-v2
        """
        self._model_name = model_name
        self._model = None

    def _load_model(self):
        """
        Lazy load the cross-encoder model on first use.
        
        This delays model loading until actually needed to reduce startup time.
        """
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                logger.info(f"Loading cross-encoder model: {self._model_name}")
                self._model = CrossEncoder(self._model_name)
                logger.info("Cross-encoder model loaded successfully")
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required. "
                    "Install it with: pip install sentence-transformers"
                )
        return self._model

    def rerank(
        self,
        query: str,
        passages: List[str],
        top_k: Optional[int] = None
    ) -> List[Dict]:
        """
        Rerank passages based on query relevance using cross-encoder.
        
        Args:
            query: Search query string
            passages: List of text passages to rerank
            top_k: Optional number of top results to return
                   If None, returns all reranked passages
        
        Returns:
            List of dicts with keys: "index", "score", "text"
            Sorted by score descending (most relevant first)
        
        Raises:
            ValueError: If query is empty or passages is empty
        """
        # Handle edge cases
        if not query or not query.strip():
            logger.warning("Empty query provided for reranking")
            return []
        
        if not passages:
            logger.warning("Empty passages list provided for reranking")
            return []
        
        # Load model
        model = self._load_model()
        
        # Prepare query-passage pairs
        pairs = [[query, passage] for passage in passages]
        
        # Batch process passages
        try:
            scores = model.predict(pairs)
        except Exception as e:
            logger.error(f"Error during cross-encoder prediction: {e}")
            raise RuntimeError(f"Reranking failed: {str(e)}")
        
        # Create results with original indices
        results = [
            {
                "index": idx,
                "score": float(score),
                "text": passage
            }
            for idx, (score, passage) in enumerate(zip(scores, passages))
        ]
        
        # Sort by score descending
        results.sort(key=lambda x: x["score"], reverse=True)
        
        # Log top scores for debugging
        if results:
            top_scores = [r["score"] for r in results[:3]]
            logger.info(f"Reranking complete. Top scores: {top_scores}")
        
        # Return top-k if specified
        if top_k is not None:
            results = results[:top_k]
        
        return results

    def rerank_with_metadata(
        self,
        query: str,
        results: List[Dict],
        top_k: Optional[int] = None,
        batch_size: int = 100
    ) -> List[Dict]:
        """
        Rerank search results with metadata preservation.
        
        Takes results from hybrid_search (or similar) and reranks them
        based on the "text" field while preserving all metadata.
        
        Args:
            query: Search query string
            results: List of result dicts from hybrid_search
                    Expected format: {"id", "text", "metadata", ...}
            top_k: Optional number of top results to return
            batch_size: Number of passages to process in each batch
                       Default: 100 (efficient for cross-encoder)
        
        Returns:
            Reranked results with all original metadata preserved
            Format: [{"id", "score", "text", "metadata", ...}]
            Sorted by rerank score descending
        
        Raises:
            ValueError: If query is empty or results is empty
        """
        # Handle edge cases
        if not query or not query.strip():
            logger.warning("Empty query provided for reranking")
            return []
        
        if not results:
            logger.warning("Empty results list provided for reranking")
            return []
        
        # Extract passages and preserve original data
        passages = [result.get("text", "") for result in results]
        
        # Batch process if many passages
        if len(passages) > batch_size:
            logger.info(f"Batch processing {len(passages)} passages in batches of {batch_size}")
            all_reranked = []
            
            for i in range(0, len(passages), batch_size):
                batch_passages = passages[i:i + batch_size]
                batch_results = results[i:i + batch_size]
                
                batch_reranked = self._rerank_batch(query, batch_passages, batch_results)
                all_reranked.extend(batch_reranked)
            
            # Sort all batches by score
            all_reranked.sort(key=lambda x: x["score"], reverse=True)
            reranked_results = all_reranked
        else:
            reranked_results = self._rerank_batch(query, passages, results)
        
        # Return top-k if specified
        if top_k is not None:
            reranked_results = reranked_results[:top_k]
        
        logger.info(f"Reranking complete. Returning {len(reranked_results)} results")
        return reranked_results

    def _rerank_batch(
        self,
        query: str,
        passages: List[str],
        results: List[Dict]
    ) -> List[Dict]:
        """
        Internal method to rerank a single batch of passages.
        
        Args:
            query: Search query string
            passages: List of text passages
            results: Original result dicts with metadata
        
        Returns:
            Reranked results with metadata preserved
        """
        # Load model
        model = self._load_model()
        
        # Prepare query-passage pairs
        pairs = [[query, passage] for passage in passages]
        
        # Get scores
        try:
            scores = model.predict(pairs)
        except Exception as e:
            logger.error(f"Error during cross-encoder prediction: {e}")
            raise RuntimeError(f"Reranking failed: {str(e)}")
        
        # Update results with rerank scores
        reranked = []
        for idx, (score, result) in enumerate(zip(scores, results)):
            reranked_result = result.copy()
            reranked_result["score"] = float(score)
            reranked_result["rerank_score"] = float(score)
            reranked.append(reranked_result)
        
        # Sort by rerank score descending
        reranked.sort(key=lambda x: x["score"], reverse=True)
        
        return reranked


@lru_cache(maxsize=1)
def get_reranker(
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
) -> RerankerService:
    """
    Get a singleton RerankerService instance.
    
    Args:
        model_name: Name of the cross-encoder model to use
    
    Returns:
        Cached RerankerService instance
    """
    return RerankerService(model_name=model_name)
