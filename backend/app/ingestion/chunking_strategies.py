"""
Chunking strategies for document ingestion.

This module provides various chunking strategies for splitting documents
into smaller pieces for embedding and retrieval.

Strategies:
- parent_child_chunking: Splits into small chunks with larger parent context
- metadata_enriched_chunking: Splits with metadata attached to each chunk

Integration with Embedding Pipeline:
```python
from backend.app.ingestion.chunking_strategies import parent_child_chunking
from backend.app.core.embeddings import get_embedding_model

# 1. Chunk document
chunks = parent_child_chunking(
    document_text=document_text,
    chunk_size=256,
    parent_size=1024
)

# 2. Embed small chunks only (faster)
embedder = get_embedding_model()
for chunk in chunks:
    chunk["embedding"] = embedder.embed_query(chunk["chunk"])

# 3. Store in vector database
vector_store.upsert_points(
    collection="documents",
    points=[
        {
            "id": chunk["id"],
            "vector": chunk["embedding"],
            "payload": {
                "text": chunk["chunk"],
                "parent": chunk["parent"],
                "metadata": chunk["metadata"]
            }
        }
        for chunk in chunks
    ]
)

# 4. On retrieval, return chunk + parent for context
results = vector_store.search(...)
for result in results:
    context = result["payload"]["parent"]  # Use parent for full context
    snippet = result["payload"]["chunk"]   # Use chunk for display
```
"""

import uuid
from typing import Dict, List


def _estimate_tokens(text: str) -> int:
    """
    Estimate token count using rough approximation.
    
    Uses the heuristic: 1 token ≈ 4 characters.
    
    Args:
        text: Input text
    
    Returns:
        Estimated token count
    """
    return len(text) // 4


def parent_child_chunking(
    document_text: str,
    chunk_size: int = 256,
    parent_size: int = 1024,
    overlap: int = 20
) -> List[Dict]:
    """
    Split document into small chunks with larger parent context.
    
    This strategy creates small chunks for efficient retrieval while
    maintaining larger parent chunks for context. When retrieving,
    you get the small chunk but can use the parent for full context.
    
    Args:
        document_text: Full document text to chunk
        chunk_size: Size of small chunks in tokens (default: 256)
        parent_size: Size of parent chunks in tokens (default: 1024)
        overlap: Token overlap between chunks (default: 20)
    
    Returns:
        List of chunk dicts with keys:
        - "id": str - unique chunk identifier
        - "chunk": str - small chunk text
        - "parent": str - larger parent context
        - "metadata": dict - chunk metadata (parent_id, chunk_index, etc.)
    
    Example:
        >>> text = "This is a long document about real estate laws..."
        >>> chunks = parent_child_chunking(text, chunk_size=256, parent_size=1024)
        >>> len(chunks)
        5
        >>> chunks[0]["chunk"]  # Small snippet
        'This is a long document...'
        >>> chunks[0]["parent"]  # Larger context
        'This is a long document about real estate laws...'
    """
    if not document_text or not document_text.strip():
        return []
    
    # Convert token sizes to character estimates
    chunk_chars = chunk_size * 4
    parent_chars = parent_size * 4
    overlap_chars = overlap * 4
    
    # First, create parent chunks
    parent_chunks = []
    start = 0
    parent_index = 0
    
    while start < len(document_text):
        end = min(start + parent_chars, len(document_text))
        parent_text = document_text[start:end]
        parent_chunks.append({
            "id": str(uuid.uuid4()),
            "text": parent_text,
            "index": parent_index,
            "start": start,
            "end": end
        })
        parent_index += 1
        start = end - overlap_chars if end < len(document_text) else end
    
    # Then, create child chunks within each parent
    all_chunks = []
    
    for parent in parent_chunks:
        parent_text = parent["text"]
        chunk_start = 0
        chunk_index = 0
        
        while chunk_start < len(parent_text):
            chunk_end = min(chunk_start + chunk_chars, len(parent_text))
            chunk_text = parent_text[chunk_start:chunk_end]
            
            all_chunks.append({
                "id": str(uuid.uuid4()),
                "chunk": chunk_text,
                "parent": parent_text,
                "metadata": {
                    "parent_id": parent["id"],
                    "parent_index": parent["index"],
                    "chunk_index": chunk_index,
                    "chunk_size": _estimate_tokens(chunk_text),
                    "parent_size": _estimate_tokens(parent_text)
                }
            })
            
            chunk_index += 1
            chunk_start = chunk_end - overlap_chars if chunk_end < len(parent_text) else chunk_end
    
    return all_chunks


def metadata_enriched_chunking(
    document_text: str,
    metadata: Dict,
    chunk_size: int = 256,
    overlap: int = 20
) -> List[Dict]:
    """
    Split document into chunks with metadata attached to each chunk.
    
    This strategy splits text into chunks and enriches each chunk with
    provided metadata. Useful for property listings, legal documents, etc.
    
    Args:
        document_text: Full document text to chunk
        metadata: Metadata dict to attach to each chunk
                 e.g., {"source": "property_listing.csv", "property_id": "prop_123"}
        chunk_size: Size of chunks in tokens (default: 256)
        overlap: Token overlap between chunks (default: 20)
    
    Returns:
        List of chunk dicts with keys:
        - "id": str - unique chunk identifier
        - "chunk": str - chunk text
        - "metadata": dict - provided metadata + chunk metadata
    
    Example:
        >>> text = "Spacious 2BHK apartment in Bangalore..."
        >>> metadata = {"source": "property_listing", "property_id": "prop_123", "city": "Bangalore"}
        >>> chunks = metadata_enriched_chunking(text, metadata, chunk_size=256)
        >>> chunks[0]["metadata"]["city"]
        'Bangalore'
        >>> chunks[0]["metadata"]["chunk_index"]
        0
    """
    if not document_text or not document_text.strip():
        return []
    
    # Convert token sizes to character estimates
    chunk_chars = chunk_size * 4
    overlap_chars = overlap * 4
    
    chunks = []
    start = 0
    chunk_index = 0
    
    while start < len(document_text):
        end = min(start + chunk_chars, len(document_text))
        chunk_text = document_text[start:end]
        
        # Create chunk metadata (copy to avoid mutation)
        chunk_metadata = metadata.copy()
        chunk_metadata.update({
            "chunk_index": chunk_index,
            "chunk_size": _estimate_tokens(chunk_text),
            "start_char": start,
            "end_char": end
        })
        
        chunks.append({
            "id": str(uuid.uuid4()),
            "chunk": chunk_text,
            "metadata": chunk_metadata
        })
        
        chunk_index += 1
        start = end - overlap_chars if end < len(document_text) else end
    
    return chunks


def simple_chunking(
    document_text: str,
    chunk_size: int = 256,
    overlap: int = 20
) -> List[Dict]:
    """
    Simple chunking without parent context or metadata enrichment.
    
    Basic chunking strategy for straightforward use cases.
    
    Args:
        document_text: Full document text to chunk
        chunk_size: Size of chunks in tokens (default: 256)
        overlap: Token overlap between chunks (default: 20)
    
    Returns:
        List of chunk dicts with keys:
        - "id": str - unique chunk identifier
        - "chunk": str - chunk text
        - "metadata": dict - basic chunk metadata
    
    Example:
        >>> text = "This is a document..."
        >>> chunks = simple_chunking(text, chunk_size=256)
        >>> len(chunks)
        3
    """
    if not document_text or not document_text.strip():
        return []
    
    # Convert token sizes to character estimates
    chunk_chars = chunk_size * 4
    overlap_chars = overlap * 4
    
    chunks = []
    start = 0
    chunk_index = 0
    
    while start < len(document_text):
        end = min(start + chunk_chars, len(document_text))
        chunk_text = document_text[start:end]
        
        chunks.append({
            "id": str(uuid.uuid4()),
            "chunk": chunk_text,
            "metadata": {
                "chunk_index": chunk_index,
                "chunk_size": _estimate_tokens(chunk_text),
                "start_char": start,
                "end_char": end
            }
        })
        
        chunk_index += 1
        start = end - overlap_chars if end < len(document_text) else end
    
    return chunks
