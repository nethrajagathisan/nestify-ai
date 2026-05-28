"""
Test script for BM25Store implementation.

Run with: python -m backend.tests.test_bm25_store
"""

from backend.app.core.bm25_store import BM25Store, get_bm25_store


def test_bm25_store():
    """Test BM25Store functionality."""
    
    # Sample documents
    documents = [
        {
            "id": "doc1",
            "text": "This is a property description with keywords like apartment, rent, and lease.",
            "metadata": {"type": "property", "category": "rental"}
        },
        {
            "id": "doc2",
            "text": "Legal FAQ: What are the tenant rights regarding security deposits?",
            "metadata": {"type": "legal", "category": "faq"}
        },
        {
            "id": "doc3",
            "text": "The apartment has three bedrooms, two bathrooms, and a kitchen.",
            "metadata": {"type": "property", "category": "description"}
        },
        {
            "id": "doc4",
            "text": "Lease agreements must be signed by both landlord and tenant.",
            "metadata": {"type": "legal", "category": "contract"}
        },
        {
            "id": "doc5",
            "text": "Rent payment is due on the first of every month.",
            "metadata": {"type": "property", "category": "payment"}
        }
    ]
    
    # Test 1: Initialize and index
    print("Test 1: Initialize and index documents")
    store = BM25Store(documents)
    assert len(store) == 5, f"Expected 5 documents, got {len(store)}"
    print("✓ Index built successfully with 5 documents")
    
    # Test 2: Query that exists exactly (should return high score)
    print("\nTest 2: Query with exact match 'apartment'")
    results = store.search("apartment", top_k=3)
    print(f"Found {len(results)} results:")
    for r in results:
        print(f"  - {r['id']}: score={r['score']:.4f}, text='{r['text'][:50]}...'")
    assert len(results) > 0, "Should return results for 'apartment'"
    assert any("apartment" in r["text"].lower() for r in results), "Results should contain 'apartment'"
    print("✓ Exact match query returns high scores")
    
    # Test 3: Query that doesn't exist (should return lower/zero scores)
    print("\nTest 3: Query with no match 'spaceship'")
    results = store.search("spaceship", top_k=3)
    print(f"Found {len(results)} results:")
    for r in results:
        print(f"  - {r['id']}: score={r['score']:.4f}")
    # BM25 may still return results with zero scores
    assert all(r['score'] == 0 for r in results), "Non-matching query should return zero scores"
    print("✓ Non-matching query returns zero scores")
    
    # Test 4: Empty query (should return empty list)
    print("\nTest 4: Empty query")
    results = store.search("", top_k=5)
    assert len(results) == 0, "Empty query should return empty list"
    print("✓ Empty query returns empty list")
    
    # Test 5: Case-insensitive search
    print("\nTest 5: Case-insensitive query 'LEASE'")
    results = store.search("LEASE", top_k=3)
    print(f"Found {len(results)} results:")
    for r in results:
        print(f"  - {r['id']}: score={r['score']:.4f}")
    assert len(results) > 0, "Case-insensitive search should work"
    print("✓ Case-insensitive search works")
    
    # Test 6: Metadata filtering
    print("\nTest 6: Metadata filtering (type='legal')")
    results = store.search("tenant", top_k=5, metadata_filter={"type": "legal"})
    print(f"Found {len(results)} results with type='legal':")
    for r in results:
        print(f"  - {r['id']}: score={r['score']:.4f}, metadata={r['metadata']}")
    assert all(r['metadata']['type'] == 'legal' for r in results), "All results should have type='legal'"
    print("✓ Metadata filtering works")
    
    # Test 7: Update document
    print("\nTest 7: Update document")
    success = store.update("doc1", "Updated text with new keywords like house and home.")
    assert success, "Update should succeed"
    updated_doc = store.get_document("doc1")
    assert "house" in updated_doc["text"], "Updated text should contain 'house'"
    print(f"✓ Document updated: {updated_doc['text'][:50]}...")
    
    # Test 8: Search after update
    print("\nTest 8: Search after update for 'house'")
    results = store.search("house", top_k=3)
    print(f"Found {len(results)} results:")
    for r in results:
        print(f"  - {r['id']}: score={r['score']:.4f}")
    assert any(r['id'] == 'doc1' for r in results), "Updated document should appear in results"
    print("✓ Search works after update")
    
    # Test 9: Singleton function
    print("\nTest 9: Singleton get_bm25_store()")
    store1 = get_bm25_store()
    store2 = get_bm25_store()
    assert store1 is store2, "Singleton should return same instance"
    print("✓ Singleton function works")
    
    # Test 10: Empty index handling
    print("\nTest 10: Empty index")
    empty_store = BM25Store()
    results = empty_store.search("test", top_k=5)
    assert len(results) == 0, "Empty index should return empty list"
    print("✓ Empty index handled correctly")
    
    print("\n" + "="*50)
    print("All tests passed! ✓")
    print("="*50)


if __name__ == "__main__":
    test_bm25_store()
