import uuid
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..config import get_settings
from ..core.embeddings import get_embedding_service
from ..core.vector_store import get_vector_store_client


def load_faq_docs() -> None:
    """Load FAQ documents from markdown files, chunk them, embed, and upsert to Qdrant."""
    settings = get_settings()
    embedding_service = get_embedding_service()
    vector_store = get_vector_store_client()
    
    # Read all .md files from data/legal/
    legal_dir = Path("data/legal")
    if not legal_dir.exists():
        raise FileNotFoundError(f"Legal directory not found: {legal_dir}")
    
    md_files = list(legal_dir.glob("*.md"))
    if not md_files:
        print(f"No .md files found in {legal_dir}")
        return
    
    print(f"Found {len(md_files)} markdown files in {legal_dir}")
    
    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80,
        separators=["\n\n", "\n", ".", " "],
    )
    
    # Process each file
    all_chunks = []
    for md_file in md_files:
        with open(md_file, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Split into chunks
        chunks = text_splitter.split_text(content)
        
        # Create human-readable title from filename
        filename = md_file.stem
        doc_title = filename.replace("_", " ").replace("-", " ").title()
        
        for chunk_index, chunk_text in enumerate(chunks):
            all_chunks.append({
                "text": chunk_text,
                "source": filename,
                "doc_title": doc_title,
                "chunk_index": chunk_index,
            })
    
    if not all_chunks:
        print("No chunks generated from documents.")
        return
    
    print(f"Generated {len(all_chunks)} chunks from {len(md_files)} documents")
    
    # Extract texts for embedding
    texts = [chunk["text"] for chunk in all_chunks]
    
    # Embed all chunks in batches
    print(f"Embedding {len(texts)} chunks...")
    embeddings = embedding_service.embed_batch(texts)
    
    # Recreate collection (idempotent)
    vector_size = len(embeddings[0]) if embeddings else 384
    print(f"Recreating collection '{settings.QDRANT_FAQ_COLLECTION}'...")
    vector_store.create_collection(
        name=settings.QDRANT_FAQ_COLLECTION,
        vector_size=vector_size,
    )
    
    # Prepare points for upsert
    points = []
    for chunk, embedding in zip(all_chunks, embeddings):
        points.append({
            "id": str(uuid.uuid4()),
            "vector": embedding,
            "payload": {
                "text": chunk["text"],
                "source": chunk["source"],
                "doc_title": chunk["doc_title"],
                "chunk_index": chunk["chunk_index"],
            },
        })
    
    # Upsert to Qdrant
    print(f"Upserting {len(points)} points to Qdrant...")
    vector_store.upsert_points(
        collection=settings.QDRANT_FAQ_COLLECTION,
        points=points,
    )
    
    print(f"Ingested {len(all_chunks)} chunks from {len(md_files)} documents")
