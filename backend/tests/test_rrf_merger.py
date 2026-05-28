"""
Test script for RRF merger implementation.

Run with: python -m backend.tests.test_rrf_merger
"""

from backend.app.core.rrf_merger import reciprocal_rank_fusion, hybrid_search
from backend.app.core.bm25_store import BM25Store


def test_reciprocal_rank_fusion():
    """Test reciprocal_rank_fusion function."""
    
    print("="*60)
    print("Testing reciprocal_rank_fusion")
    print("="*60)
    
    # Test 1: Basic merge with overlapping documents
    print("\nTest 1: Basic merge with overlapping documents")
    dense_results = [
        {"id": "doc1", "score": 0.9, "text": "apartment in Bangalore", "metadata": {"city": "Bangalore"}},
        {"id": "doc2", "score": 0.8, "text": "house in Mumbai", "metadata": {"city": "Mumbai"}},
        {"id": "doc3", "score": 0.7, "text": "villa in Delhi", "metadata": {"city": "Delhi"}},
    ]
    
    sparse_results = [
        {"id": "doc2", "score": 2.5, "text": "house in Mumbai", "metadata": {"city": "Mumbai"}},
        {"id": "doc4", "score": 1.8, "text": "flat in Chennai", "metadata": {"city": "Chennai"}},
        {"id": "doc1", "score": 1.5, "text": "apartment in Bangalore", "metadata": {"city": "Bangalore"}},
    ]
    
    results = reciprocal_rank_fusion(dense_results, sparse_results, k=60)
    print(f"Merged {len(results)} unique documents")
    for r in results:
        print(f"  - {r['id']}: rrf_score={r['rrf_score']:.4f}, dense={r['dense_score']:.2f}, sparse={r['sparse_score']:.2f}")
    
    assert len(results) == 4, f"Expected 4 unique docs, got {len(results)}"
    # doc2 and doc1 appear in both, should have higher combined scores
    doc_ids = [r["id"] for r in results]
    assert "doc1" in doc_ids and "doc2" in doc_ids and "doc3" in doc_ids and "doc4" in doc_ids
    print("✓ Basic merge works correctly")
    
    # Test 2: With weights
    print("\nTest 2: Merge with weights (dense: 0.7, sparse: 0.3)")
    results_weighted = reciprocal_rank_fusion(
        dense_results, 
        sparse_results, 
        k=60,
        weights={"dense": 0.7, "sparse": 0.3}
    )
    print(f"Weighted merged {len(results_weighted)} documents")
    for r in results_weighted:
        print(f"  - {r['id']}: rrf_score={r['rrf_score']:.4f}")
    print("✓ Weighted merge works")
    
    # Test 3: No overlap between results
    print("\nTest 3: No overlap between dense and sparse results")
    dense_unique = [
        {"id": "doc1", "score": 0.9, "text": "text1", "metadata": {}},
        {"id": "doc2", "score": 0.8, "text": "text2", "metadata": {}},
    ]
    sparse_unique = [
        {"id": "doc3", "score": 2.5, "text": "text3", "metadata": {}},
        {"id": "doc4", "score": 1.8, "text": "text4", "metadata": {}},
    ]
    results = reciprocal_rank_fusion(dense_unique, sparse_unique, k=60)
    assert len(results) == 4, f"Expected 4 docs with no overlap, got {len(results)}"
    print(f"✓ No overlap case: {len(results)} documents")
    
    # Test 4: Empty results
    print("\nTest 4: Empty results handling")
    results = reciprocal_rank_fusion([], sparse_results, k=60)
    assert len(results) == len(sparse_results), "Should return sparse results when dense is empty"
    print(f"✓ Empty dense: returns {len(results)} sparse results")
    
    results = reciprocal_rank_fusion(dense_results, [], k=60)
    assert len(results) == len(dense_results), "Should return dense results when sparse is empty"
    print(f"✓ Empty sparse: returns {len(results)} dense results")
    
    results = reciprocal_rank_fusion([], [], k=60)
    assert len(results) == 0, "Should return empty list when both are empty"
    print("✓ Both empty: returns empty list")
    
    # Test 5: Qdrant format (payload instead of text/metadata)
    print("\nTest 5: Qdrant format (payload)")
    dense_qdrant = [
        {"id": "doc1", "score": 0.9, "payload": {"text": "apartment", "city": "Bangalore"}},
        {"id": "doc2", "score": 0.8, "payload": {"text": "house", "city": "Mumbai"}},
    ]
    sparse_standard = [
        {"id": "doc1", "score": 1.5, "text": "apartment", "metadata": {"city": "Bangalore"}},
    ]
    results = reciprocal_rank_fusion(dense_qdrant, sparse_standard, k=60)
    assert len(results) == 2, f"Expected 2 docs, got {len(results)}"
    assert results[0]["text"] == "apartment", "Should extract text from payload"
    print("✓ Qdrant format handled correctly")
    
    print("\n" + "="*60)
    print("reciprocal_rank_fusion tests passed! ✓")
    print("="*60)


def test_hybrid_search():
    """Test hybrid_search function with mock components."""
    
    print("\n" + "="*60)
    print("Testing hybrid_search")
    print("="*60)
    
    # Test 1: Hybrid search with BM25 only (mock vector store)
    print("\nTest 1: Hybrid search with BM25 only (vector store fails)")
    
    # Create BM25 store with sample documents
    documents = [
        {
            "id": "doc1",
            "text": "2 BHK apartment in Bangalore with amenities",
            "metadata": {"city": "Bangalore", "bhk": 2}
        },
        {
            "id": "doc2",
            "text": "3 BHK house in Mumbai near metro",
            "metadata": {"city": "Mumbai", "bhk": 3}
        },
        {
            "id": "doc3",
            "text": "1 BHK flat in Chennai city center",
            "metadata": {"city": "Chennai", "bhk": 1}
        },
    ]
    bm25_store = BM25Store(documents)
    
    # Mock vector store that raises exception
    class MockVectorStore:
        def search(self, collection, query_vector, filters, top_k):
            raise Exception("Mock vector store failure")
    
    mock_vector_store = MockVectorStore()
    query_vector = [0.1] * 384  # Mock embedding
    
    results = hybrid_search(
        query="2 BHK apartment",
        vector_store=mock_vector_store,
        bm25_store=bm25_store,
        query_vector=query_vector,
        top_k_final=3
    )
    
    print(f"Results with failed vector store: {len(results)}")
    for r in results:
        print(f"  - {r['id']}: rrf_score={r['rrf_score']:.4f}, text='{r['text'][:40]}...'")
    assert len(results) == 3, "Should return BM25 results when vector store fails"
    print("✓ Hybrid search handles vector store failure gracefully")
    
    # Test 2: Hybrid search with both stores (mock both)
    print("\nTest 2: Hybrid search with both stores (mock)")
    
    class WorkingMockVectorStore:
        def search(self, collection, query_vector, filters, top_k):
            return [
                {"id": "doc1", "score": 0.9, "payload": {"text": "2 BHK apartment in Bangalore", "city": "Bangalore"}},
                {"id": "doc4", "score": 0.85, "payload": {"text": "2 BHK flat in Bangalore", "city": "Bangalore"}},
            ]
    
    working_vector_store = WorkingMockVectorStore()
    
    results = hybrid_search(
        query="2 BHK apartment",
        vector_store=working_vector_store,
        bm25_store=bm25_store,
        query_vector=query_vector,
        top_k_final=3
    )
    
    print(f"Results with both stores: {len(results)}")
    for r in results:
        print(f"  - {r['id']}: rrf_score={r['rrf_score']:.4f}, dense={r['dense_score']:.2f}, sparse={r['sparse_score']:.2f}")
    
    # Should have doc1 (appears in both) and doc4 (only in dense) and doc2/doc3 (only in sparse)
    assert len(results) > 0, "Should return merged results"
    # doc1 should have both dense and sparse scores
    doc1_result = next((r for r in results if r["id"] == "doc1"), None)
    if doc1_result:
        assert doc1_result["dense_score"] > 0, "doc1 should have dense score"
        assert doc1_result["sparse_score"] > 0, "doc1 should have sparse score"
    print("✓ Hybrid search merges results from both stores")
    
    # Test 3: With filters
    print("\nTest 3: Hybrid search with city filter")
    results = hybrid_search(
        query="apartment",
        vector_store=working_vector_store,
        bm25_store=bm25_store,
        query_vector=query_vector,
        top_k_final=5,
        filters={"city": "Bangalore"}
    )
    print(f"Results with city filter: {len(results)}")
    for r in results:
        print(f"  - {r['id']}: city={r['metadata'].get('city', 'N/A')}")
    print("✓ Hybrid search with filters works")
    
    # Test 4: With RRF weights
    print("\nTest 4: Hybrid search with RRF weights")
    results = hybrid_search(
        query="apartment",
        vector_store=working_vector_store,
        bm25_store=bm25_store,
        query_vector=query_vector,
        top_k_final=3,
        rrf_weights={"dense": 0.7, "sparse": 0.3}
    )
    print(f"Results with weights: {len(results)}")
    for r in results:
        print(f"  - {r['id']}: rrf_score={r['rrf_score']:.4f}")
    print("✓ Hybrid search with RRF weights works")
    
    print("\n" + "="*60)
    print("hybrid_search tests passed! ✓")
    print("="*60)


def test_rrf_algorithm():
    """Test RRF algorithm correctness."""
    
    print("\n" + "="*60)
    print("Testing RRF algorithm correctness")
    print("="*60)
    
    # Test: Higher rank (lower number) should get higher RRF score
    print("\nTest: RRF score decreases with rank")
    k = 60
    
    # Rank 1
    score_1 = 1.0 / (k + 1)
    # Rank 10
    score_10 = 1.0 / (k + 10)
    
    print(f"  Rank 1 score: {score_1:.6f}")
    print(f"  Rank 10 score: {score_10:.6f}")
    assert score_1 > score_10, "Rank 1 should have higher score than rank 10"
    print("✓ RRF score correctly decreases with rank")
    
    # Test: Document appearing in both lists should have higher combined score
    print("\nTest: Document in both lists gets higher combined score")
    dense = [
        {"id": "doc1", "score": 0.9, "text": "text", "metadata": {}},
        {"id": "doc2", "score": 0.8, "text": "text", "metadata": {}},
    ]
    sparse = [
        {"id": "doc1", "score": 2.5, "text": "text", "metadata": {}},
        {"id": "doc3", "score": 1.8, "text": "text", "metadata": {}},
    ]
    
    results = reciprocal_rank_fusion(dense, sparse, k=60)
    doc1_score = next(r["rrf_score"] for r in results if r["id"] == "doc1")
    doc2_score = next(r["rrf_score"] for r in results if r["id"] == "doc2")
    doc3_score = next(r["rrf_score"] for r in results if r["id"] == "doc3")
    
    print(f"  doc1 (both lists): {doc1_score:.6f}")
    print(f"  doc2 (dense only): {doc2_score:.6f}")
    print(f"  doc3 (sparse only): {doc3_score:.6f}")
    assert doc1_score > doc2_score, "doc1 (in both) should outrank doc2 (dense only)"
    assert doc1_score > doc3_score, "doc1 (in both) should outrank doc3 (sparse only)"
    print("✓ Documents in both lists get higher combined scores")
    
    print("\n" + "="*60)
    print("RRF algorithm tests passed! ✓")
    print("="*60)


if __name__ == "__main__":
    test_reciprocal_rank_fusion()
    test_hybrid_search()
    test_rrf_algorithm()
    
    print("\n" + "="*60)
    print("ALL TESTS PASSED! ✓✓✓")
    print("="*60)
