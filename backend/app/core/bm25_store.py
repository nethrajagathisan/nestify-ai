"""
BM25Store for keyword-based retrieval using BM25 ranking.

This module provides a BM25Store class that implements BM25 algorithm
for efficient keyword-based document retrieval. It uses the rank_bm25 library
to build and search an index of documents.

Tests:
- Query that exists exactly → returns high score
- Query that doesn't exist → returns lower/zero scores
- Empty query → returns empty list
"""

import re
from functools import lru_cache
from typing import Dict, List, Optional

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    raise ImportError(
        "rank_bm25 is required. Install it with: pip install rank_bm25"
    )


class BM25Store:
    """
    BM25-based keyword retrieval store.
    
    Uses BM25Okapi algorithm for ranking documents based on keyword matches.
    Supports document indexing, searching, and updating with metadata.
    """

    def __init__(self, documents: Optional[List[Dict]] = None):
        """
        Initialize BM25Store with optional documents.
        
        Args:
            documents: List of dicts with keys: "id", "text", "metadata"
        """
        self._documents: Dict[str, Dict] = {}
        self._tokenized_docs: List[List[str]] = []
        self._bm25_index: Optional[BM25Okapi] = None
        
        if documents:
            self.index(documents)

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text using whitespace and punctuation split.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of lowercase tokens
        """
        # Split on whitespace and punctuation, convert to lowercase
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens

    def index(self, documents: List[Dict]) -> None:
        """
        Build BM25 index from documents.
        
        Args:
            documents: List of dicts with keys: "id", "text", "metadata"
                      - id: str - unique document identifier
                      - text: str - document content
                      - metadata: dict - optional metadata for filtering
        """
        self._documents.clear()
        self._tokenized_docs.clear()
        
        for doc in documents:
            doc_id = doc.get("id")
            text = doc.get("text", "")
            metadata = doc.get("metadata", {})
            
            if doc_id is None:
                continue
                
            self._documents[doc_id] = {
                "text": text,
                "metadata": metadata
            }
            self._tokenized_docs.append(self._tokenize(text))
        
        # Build BM25 index if we have documents
        if self._tokenized_docs:
            self._bm25_index = BM25Okapi(self._tokenized_docs)
        else:
            self._bm25_index = None

    def search(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Search for documents matching the query using BM25 ranking.
        
        Args:
            query: Search query string
            top_k: Number of top results to return
            metadata_filter: Optional dict to filter results by metadata
            
        Returns:
            List of dicts with keys: "id", "score", "text", "metadata"
            Returns empty list if query is empty or index is empty.
        """
        # Handle empty query
        if not query or not query.strip():
            return []
        
        # Handle empty index
        if self._bm25_index is None or not self._tokenized_docs:
            return []
        
        # Tokenize query
        tokenized_query = self._tokenize(query)
        
        # Get BM25 scores
        scores = self._bm25_index.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]
        
        # Build results
        results = []
        doc_ids = list(self._documents.keys())
        
        for idx in top_indices:
            doc_id = doc_ids[idx]
            doc_data = self._documents[doc_id]
            
            # Apply metadata filter if provided
            if metadata_filter:
                doc_metadata = doc_data.get("metadata", {})
                if not all(
                    doc_metadata.get(k) == v
                    for k, v in metadata_filter.items()
                ):
                    continue
            
            results.append({
                "id": doc_id,
                "score": float(scores[idx]),
                "text": doc_data["text"],
                "metadata": doc_data["metadata"]
            })
        
        return results

    def update(self, doc_id: str, new_text: str) -> bool:
        """
        Update a document's text in the index.
        
        Args:
            doc_id: ID of the document to update
            new_text: New text content for the document
            
        Returns:
            True if update was successful, False if doc_id not found
        """
        if doc_id not in self._documents:
            return False
        
        # Update the document text
        self._documents[doc_id]["text"] = new_text
        
        # Rebuild the index (BM25 requires re-indexing)
        documents = [
            {
                "id": doc_id,
                "text": data["text"],
                "metadata": data["metadata"]
            }
            for doc_id, data in self._documents.items()
        ]
        self.index(documents)
        
        return True

    def get_document(self, doc_id: str) -> Optional[Dict]:
        """
        Retrieve a document by ID.
        
        Args:
            doc_id: Document identifier
            
        Returns:
            Document dict with "text" and "metadata", or None if not found
        """
        return self._documents.get(doc_id)

    def __len__(self) -> int:
        """Return the number of documents in the index."""
        return len(self._documents)


@lru_cache(maxsize=1)
def get_bm25_store() -> BM25Store:
    """
    Get a singleton BM25Store instance.
    
    Returns:
        Cached BM25Store instance
    """
    return BM25Store()
