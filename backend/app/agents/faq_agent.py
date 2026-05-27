from collections import defaultdict
from typing import Optional

from ..config import get_settings
from ..core.embeddings import get_embedding_service
from ..core.vector_store import get_vector_store_client
from ..core.llm import get_groq_client
from ..prompts.legal_faq import LEGAL_FAQ_SYSTEM_PROMPT


class FAQAgent:
    def __init__(self):
        self.embedding_service = get_embedding_service()
        self.vector_store = get_vector_store_client()
        self.groq_client = get_groq_client()
        self.settings = get_settings()

    def answer(self, question: str) -> dict:
        """Answer a legal FAQ question using retrieved context."""
        try:
            # Step 1: Embed the question
            question_embedding = self.embedding_service.embed_single(question)
            
            # Step 2: Search legal_faq collection
            search_results = self.vector_store.search(
                collection=self.settings.QDRANT_FAQ_COLLECTION,
                query_vector=question_embedding,
                filters=None,
                top_k=4,
            )
            
            # Step 3: Deduplicate chunks (keep 1-2 per source)
            chunks_by_source = defaultdict(list)
            for hit in search_results:
                source = hit["payload"].get("source", "unknown")
                chunks_by_source[source].append(hit)
            
            # Keep at most 2 chunks per source
            deduplicated_chunks = []
            for source, chunks in chunks_by_source.items():
                deduplicated_chunks.extend(chunks[:2])
            
            # Step 4: Build context
            context_parts = ["Relevant legal context:"]
            for hit in deduplicated_chunks:
                source = hit["payload"].get("source", "unknown")
                text = hit["payload"].get("text", "")
                context_parts.append(f"[Source: {source}]")
                context_parts.append(text)
                context_parts.append("")  # Empty line between chunks
            
            context = "\n".join(context_parts)
            
            # Step 5: Call Groq for answer
            user_message = f"Question: {question}\n\n{context}"
            answer = self.groq_client.chat(
                messages=[{"role": "user", "content": user_message}],
                system_prompt=LEGAL_FAQ_SYSTEM_PROMPT,
            )
            
            # Step 6: Extract unique source names
            sources = list(set(hit["payload"].get("source", "unknown") for hit in deduplicated_chunks))
            
            # Step 7: Return response dict
            return {
                "question": question,
                "answer": answer,
                "sources": sources,
                "chunks_retrieved": len(deduplicated_chunks),
            }
            
        except Exception as e:
            raise RuntimeError(f"Error during FAQ answering: {str(e)}")
