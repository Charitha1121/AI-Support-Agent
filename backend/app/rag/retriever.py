import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.rag.documents import DocumentChunk, chunk_documents, get_default_knowledge_base_dir
from app.rag.embeddings import (
    BaseEmbeddingProvider,
    DeterministicSemanticEmbeddingProvider,
    OpenAIEmbeddingProvider,
    cosine_similarity,
    get_embedding_provider,
)


@dataclass
class RetrievedDocument:
    document_text: str
    metadata: Dict[str, Any]
    similarity_score: float
    chunk_id: str
    title: str
    section: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_text": self.document_text,
            "metadata": self.metadata,
            "similarity_score": round(self.similarity_score, 4),
            "chunk_id": self.chunk_id,
            "title": self.title,
            "section": self.section,
        }


class KnowledgeRetriever:
    """
    Vector retrieval engine for NovaTech knowledge base documents.
    Indexes semantic chunks and performs cosine similarity search.
    """
    def __init__(
        self,
        kb_dir: Optional[Path] = None,
        embedding_provider: Optional[BaseEmbeddingProvider] = None,
    ):
        self.kb_dir = kb_dir or get_default_knowledge_base_dir()
        self.embedding_provider = embedding_provider or get_embedding_provider()
        self.chunks: List[DocumentChunk] = []
        self.chunk_embeddings: List[List[float]] = []
        self._is_indexed = False

    def build_index(self, force_rebuild: bool = False):
        """Load documents, generate chunks, and compute vector embeddings."""
        if self._is_indexed and not force_rebuild:
            return

        self.chunks = chunk_documents(self.kb_dir)
        if not self.chunks:
            self.chunk_embeddings = []
            self._is_indexed = True
            return

        texts_to_embed = [chunk.content for chunk in self.chunks]
        self.chunk_embeddings = self.embedding_provider.embed_documents(texts_to_embed)
        self._is_indexed = True

    def retrieve(
        self,
        query: str,
        k: int = 4,
        score_threshold: float = 0.15
    ) -> List[RetrievedDocument]:
        """
        Retrieve the top-k most relevant document chunks for a given query string.
        """
        if not self._is_indexed:
            self.build_index()

        if not query or not query.strip() or not self.chunks:
            return []

        query_embedding = self.embedding_provider.embed_text(query.strip())

        scored_results: List[tuple[float, DocumentChunk]] = []
        for embedding, chunk in zip(self.chunk_embeddings, self.chunks):
            score = cosine_similarity(query_embedding, embedding)
            if score >= score_threshold:
                scored_results.append((score, chunk))

        # Sort by similarity score in descending order
        scored_results.sort(key=lambda x: x[0], reverse=True)

        top_k = scored_results[:k]

        return [
            RetrievedDocument(
                document_text=chunk.content,
                metadata=chunk.metadata,
                similarity_score=score,
                chunk_id=chunk.chunk_id,
                title=chunk.title,
                section=chunk.section,
            )
            for score, chunk in top_k
        ]


# Singleton instance for application-wide retrieval
_retriever_instance: Optional[KnowledgeRetriever] = None


def get_retriever(force_reinit: bool = False) -> KnowledgeRetriever:
    """
    Retrieve or initialize the global KnowledgeRetriever singleton.
    Safely invalidates the index if the underlying provider configuration changes.
    """
    global _retriever_instance

    current_provider_config = os.getenv("EMBEDDING_PROVIDER", "local").strip().lower()

    if _retriever_instance is not None:
        # Check if provider type matches current environment configuration
        is_local_provider = isinstance(_retriever_instance.embedding_provider, DeterministicSemanticEmbeddingProvider)
        is_openai_provider = isinstance(_retriever_instance.embedding_provider, OpenAIEmbeddingProvider)

        if (
            force_reinit
            or (current_provider_config == "local" and not is_local_provider)
            or (current_provider_config == "openai" and not is_openai_provider)
        ):
            _retriever_instance = None

    if _retriever_instance is None:
        _retriever_instance = KnowledgeRetriever()
        _retriever_instance.build_index()

    return _retriever_instance


def retrieve_documents(
    query: str,
    k: int = 4,
    score_threshold: float = 0.15
) -> List[RetrievedDocument]:
    """
    Convenience function to perform knowledge retrieval using the shared singleton retriever.
    """
    retriever = get_retriever()
    return retriever.retrieve(query, k=k, score_threshold=score_threshold)
