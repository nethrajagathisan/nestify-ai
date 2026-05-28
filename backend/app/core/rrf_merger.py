"""
Reciprocal Rank Fusion (RRF) for merging dense and sparse retrieval results.

This module implements RRF algorithm to combine results from vector search (dense)
and BM25 keyword search (sparse) for hybrid retrieval.

RRF Algorithm:
- For each result in a ranked list, calculate: score = 1 / (k + rank)
- k is a constant (typically 60) that smooths the contribution of ranks
- Higher ranks (smaller rank number) get higher scores
- Merge scores from multiple lists by summing for same document
- Re-rank by combined score

Tests (unit test examples):
>>> dense = [{"id": "1", "score": 0.9, "text": "...", "metadata": {}}]
>>> sparse = [{"id": "1", "score": 2.5, "text": "...", "metadata": {}}]
>>> results = reciprocal_rank_fusion(dense, sparse, k=60)
>>> len(results) == 1
True
>>> results[0]["id"] == "1"
True
"""

from typing import Dict, List, Optional, Any

from .vector_store import VectorStoreClient
from .bm25_store import BM25Store


def reciprocal_rank_fusion(
    dense_results: List[Dict],
    sparse_results: List[Dict],
    k: int = 60,
    weights: Optional[Dict[str, float]] = None
) -> List[Dict]:
    """
    Merge dense and sparse retrieval results using Reciprocal Rank Fusion.
    
    RRF combines multiple ranked lists by converting ranks to scores:
    score = 1 / (k + rank), where k is a smoothing constant.
    
    Args:
        dense_results: Results from vector search (Qdrant)
                      Format: [{"id", "score", "text", "metadata"}]
                      or [{"id", "score", "payload"}] from Qdrant
        sparse_results: Results from BM25 keyword search
                       Format: [{"id", "score", "text", "metadata"}]
        k: RRF constant (default 60). Higher k gives more weight to lower ranks.
        weights: Optional weights for each retrieval type
                Format: {"dense": 0.6, "sparse": 0.4}
                If None, equal weights (1.0) are used.
    
    Returns:
        Merged and sorted results with combined RRF scores.
        Format: [{"id", "rrf_score", "text", "metadata", "dense_score", "sparse_score"}]
        Deduplicated by document ID.
    
    Example:
        >>> dense = [
        ...     {"id": "doc1", "score": 0.9, "text": "apartment", "metadata": {}},
        ...     {"id": "doc2", "score": 0.8, "text": "house", "metadata": {}}
        ... ]
        >>> sparse = [
        ...     {"id": "doc2", "score": 2.5, "text": "house", "metadata": {}},
        ...     {"id": "doc3", "score": 1.8, "text": "villa", "metadata": {}}
        ... ]
        >>> results = reciprocal_rank_fusion(dense, sparse, k=60)
        >>> len(results)  # doc1, doc2, doc3
        3
        >>> # doc2 appears in both, should have higher combined score
        >>> doc2_result = next(r for r in results if r["id"] == "doc2")
        >>> doc2_result["rrf_score"] > 0
        True
    """
    # Set default weights if not provided
    if weights is None:
        weights = {"dense": 1.0, "sparse": 1.0}
    
    # Calculate RRF scores for dense results
    dense_rrf = {}
    for rank, result in enumerate(dense_results, start=1):
        doc_id = result.get("id")
        if doc_id is None:
            continue
        
        rrf_score = 1.0 / (k + rank)
        dense_rrf[doc_id] = {
            "rrf_score": rrf_score * weights.get("dense", 1.0),
            "dense_score": result.get("score", 0.0),
            "dense_rank": rank,
            "result": result
        }
    
    # Calculate RRF scores for sparse results
    sparse_rrf = {}
    for rank, result in enumerate(sparse_results, start=1):
        doc_id = result.get("id")
        if doc_id is None:
            continue
        
        rrf_score = 1.0 / (k + rank)
        sparse_rrf[doc_id] = {
            "rrf_score": rrf_score * weights.get("sparse", 1.0),
            "sparse_score": result.get("score", 0.0),
            "sparse_rank": rank,
            "result": result
        }
    
    # Merge by document ID, summing RRF scores
    merged: Dict[str, Dict] = {}
    
    # Add all dense results
    for doc_id, data in dense_rrf.items():
        merged[doc_id] = {
            "id": doc_id,
            "rrf_score": data["rrf_score"],
            "dense_score": data["dense_score"],
            "dense_rank": data["dense_rank"],
            "sparse_score": 0.0,
            "sparse_rank": None,
            "result": data["result"]
        }
    
    # Add sparse results, merging with existing if present
    for doc_id, data in sparse_rrf.items():
        if doc_id in merged:
            # Document appears in both lists - sum RRF scores
            merged[doc_id]["rrf_score"] += data["rrf_score"]
            merged[doc_id]["sparse_score"] = data["sparse_score"]
            merged[doc_id]["sparse_rank"] = data["sparse_rank"]
            # Prefer sparse result for text/metadata if available
            if "text" in data["result"]:
                merged[doc_id]["result"] = data["result"]
        else:
            # Document only in sparse results
            merged[doc_id] = {
                "id": doc_id,
                "rrf_score": data["rrf_score"],
                "dense_score": 0.0,
                "dense_rank": None,
                "sparse_score": data["sparse_score"],
                "sparse_rank": data["sparse_rank"],
                "result": data["result"]
            }
    
    # Sort by combined RRF score (descending)
    sorted_results = sorted(
        merged.values(),
        key=lambda x: x["rrf_score"],
        reverse=True
    )
    
    # Format output to match expected structure
    formatted_results = []
    for item in sorted_results:
        result = item["result"]
        
        # Handle both Qdrant format (payload) and standard format (text, metadata)
        if "payload" in result:
            # Qdrant format
            formatted = {
                "id": item["id"],
                "rrf_score": item["rrf_score"],
                "text": result["payload"].get("text", ""),
                "metadata": result["payload"],
                "dense_score": item["dense_score"],
                "sparse_score": item["sparse_score"]
            }
        else:
            # Standard format
            formatted = {
                "id": item["id"],
                "rrf_score": item["rrf_score"],
                "text": result.get("text", ""),
                "metadata": result.get("metadata", {}),
                "dense_score": item["dense_score"],
                "sparse_score": item["sparse_score"]
            }
        
        formatted_results.append(formatted)
    
    return formatted_results


def hybrid_search(
    query: str,
    vector_store: VectorStoreClient,
    bm25_store: BM25Store,
    query_vector: List[float],
    collection: str = "properties",
    top_k_dense: int = 20,
    top_k_sparse: int = 20,
    top_k_final: int = 5,
    filters: Optional[Dict] = None,
    rrf_k: int = 60,
    rrf_weights: Optional[Dict[str, float]] = None
) -> List[Dict]:
    """
    Perform hybrid search combining dense vector search and BM25 keyword search.
    
    This function executes both retrieval methods in parallel and merges results
    using Reciprocal Rank Fusion for optimal relevance.
    
    Args:
        query: Search query string
        vector_store: VectorStoreClient instance for dense search
        bm25_store: BM25Store instance for sparse search
        query_vector: Query embedding vector for dense search
        collection: Qdrant collection name (default "properties")
        top_k_dense: Number of results to retrieve from dense search
        top_k_sparse: Number of results to retrieve from sparse search
        top_k_final: Number of final results to return after RRF merging
        filters: Optional filters for dense search (Qdrant format)
                e.g., {"city": "Bangalore", "bhk": 2}
        rrf_k: RRF constant for score calculation (default 60)
        rrf_weights: Optional weights for dense/sparse contributions
                     e.g., {"dense": 0.6, "sparse": 0.4}
    
    Returns:
        Top-k merged results with combined RRF scores.
        Format: [{"id", "rrf_score", "text", "metadata", "dense_score", "sparse_score"}]
    
    Example:
        >>> from backend.app.core.vector_store import get_vector_store_client
        >>> from backend.app.core.bm25_store import get_bm25_store
        >>> from backend.app.core.embeddings import get_embedding_model
        >>> 
        >>> vector_client = get_vector_store_client()
        >>> bm25 = get_bm25_store()
        >>> embedder = get_embedding_model()
        >>> 
        >>> query = "2 BHK apartment in Bangalore"
        >>> query_vec = embedder.embed_query(query)
        >>> 
        >>> results = hybrid_search(
        ...     query=query,
        ...     vector_store=vector_client,
        ...     bm25_store=bm25,
        ...     query_vector=query_vec,
        ...     top_k_final=5,
        ...     filters={"city": "Bangalore"}
        ... )
        >>> len(results)
        5
        >>> results[0]["rrf_score"] > 0
        True
    """
    # Perform dense vector search
    try:
        dense_results = vector_store.search(
            collection=collection,
            query_vector=query_vector,
            filters=filters,
            top_k=top_k_dense
        )
    except Exception as e:
        print(f"Dense search failed: {e}")
        dense_results = []
    
    # Perform sparse BM25 search
    try:
        # Convert Qdrant filters to BM25 metadata filter if applicable
        metadata_filter = None
        if filters and "city" in filters:
            metadata_filter = {"city": filters["city"]}
        
        sparse_results = bm25_store.search(
            query=query,
            top_k=top_k_sparse,
            metadata_filter=metadata_filter
        )
    except Exception as e:
        print(f"Sparse search failed: {e}")
        sparse_results = []
    
    # Merge results using RRF
    merged_results = reciprocal_rank_fusion(
        dense_results=dense_results,
        sparse_results=sparse_results,
        k=rrf_k,
        weights=rrf_weights
    )
    
    # Return top-k final results
    return merged_results[:top_k_final]
