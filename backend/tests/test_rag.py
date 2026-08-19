import math
import os
import sys
from pathlib import Path
from unittest.mock import patch

# Ensure backend root is on sys.path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.rag.documents import chunk_documents, get_default_knowledge_base_dir, load_raw_documents
from app.rag.embeddings import (
    DeterministicSemanticEmbeddingProvider,
    OpenAIEmbeddingProvider,
    cosine_similarity,
    get_embedding_provider,
)
from app.rag.retriever import KnowledgeRetriever, get_retriever, retrieve_documents


def test_document_loading():
    """Verify that all 10 knowledge base markdown documents are loaded correctly."""
    kb_dir = get_default_knowledge_base_dir()
    raw_docs = load_raw_documents(kb_dir)
    assert len(raw_docs) == 10, f"Expected 10 documents, got {len(raw_docs)}"

    filenames = [doc["filename"] for doc in raw_docs]
    assert "01_refund_policy.md" in filenames
    assert "02_shipping_and_delivery.md" in filenames
    assert "03_subscription_tiers.md" in filenames
    assert "04_cancellation_policy.md" in filenames
    assert "05_payment_methods.md" in filenames
    assert "06_account_security.md" in filenames
    assert "07_order_cancellation.md" in filenames
    assert "08_delivery_issues.md" in filenames
    assert "09_subscription_renewal.md" in filenames
    assert "10_contact_and_support.md" in filenames
    print(f"PASS: Document loading ({len(raw_docs)} documents verified)")


def test_document_chunking():
    """Verify that documents are parsed into structured semantic chunks with metadata (33 chunks)."""
    chunks = chunk_documents()
    assert len(chunks) == 33, f"Expected 33 chunks, got {len(chunks)}"

    first_chunk = chunks[0]
    assert hasattr(first_chunk, "chunk_id")
    assert hasattr(first_chunk, "title")
    assert hasattr(first_chunk, "section")
    assert hasattr(first_chunk, "content")
    assert "doc_id" in first_chunk.metadata
    print(f"PASS: Semantic chunking ({len(chunks)} chunks produced)")


def test_local_provider_selection():
    """Verify that EMBEDDING_PROVIDER=local selects DeterministicSemanticEmbeddingProvider."""
    with patch.dict(os.environ, {"EMBEDDING_PROVIDER": "local"}):
        provider = get_embedding_provider()
        assert isinstance(provider, DeterministicSemanticEmbeddingProvider)
        assert provider.dimensions == 256
    print("PASS: Local embedding provider selection")


def test_deterministic_embedding_reproducibility():
    """Verify that embeddings for identical text are 100% reproducible across calls."""
    provider = DeterministicSemanticEmbeddingProvider()
    text = "Customers may request a refund within 14 days of purchase."
    vec1 = provider.embed_text(text)
    vec2 = provider.embed_text(text)
    assert vec1 == vec2, "Vectors for identical text must be identical"
    print("PASS: Deterministic embedding reproducibility")


def test_different_text_produces_different_vectors():
    """Verify that distinct text inputs produce different vectors."""
    provider = DeterministicSemanticEmbeddingProvider()
    vec1 = provider.embed_text("What is your refund policy?")
    vec2 = provider.embed_text("How long does express shipping take?")
    assert vec1 != vec2, "Different texts should produce different vectors"
    similarity = cosine_similarity(vec1, vec2)
    assert similarity < 1.0, f"Expected similarity < 1.0, got {similarity}"
    print("PASS: Distinct text vector separation")


def test_vector_normalization():
    """Verify that non-empty text vectors have 256 dimensions and unit norm (~1.0)."""
    provider = DeterministicSemanticEmbeddingProvider()
    vec = provider.embed_text("NovaTech cloud storage and subscription plans.")
    assert len(vec) == 256, f"Expected 256 dimensions, got {len(vec)}"
    norm = math.sqrt(sum(v * v for v in vec))
    assert abs(norm - 1.0) < 1e-5, f"Expected unit norm ~1.0, got {norm}"
    print("PASS: Vector normalization")


def test_refund_policy_retrieval():
    """Test query: 'What's your refund policy?'"""
    with patch.dict(os.environ, {"EMBEDDING_PROVIDER": "local"}):
        retriever = get_retriever(force_reinit=True)
        results = retriever.retrieve("What's your refund policy?", k=3)
        assert len(results) > 0, "No documents retrieved for refund query"
        top_doc = results[0]
        assert "refund" in top_doc.document_text.lower() or "refund" in top_doc.title.lower()
        assert top_doc.similarity_score > 0.2
        print(f"PASS: Refund policy retrieval (Top: '{top_doc.title} - {top_doc.section}', Score: {top_doc.similarity_score:.4f})")


def test_shipping_query_retrieval():
    """Test query: 'How long does shipping take?'"""
    with patch.dict(os.environ, {"EMBEDDING_PROVIDER": "local"}):
        retriever = get_retriever(force_reinit=True)
        results = retriever.retrieve("How long does shipping take?", k=3)
        assert len(results) > 0, "No documents retrieved for shipping query"
        top_doc = results[0]
        assert "ship" in top_doc.document_text.lower() or "deliver" in top_doc.document_text.lower()
        assert top_doc.similarity_score > 0.2
        print(f"PASS: Shipping query retrieval (Top: '{top_doc.title} - {top_doc.section}', Score: {top_doc.similarity_score:.4f})")


def test_subscription_query_retrieval():
    """Test query: 'What subscription plans are available?'"""
    with patch.dict(os.environ, {"EMBEDDING_PROVIDER": "local"}):
        retriever = get_retriever(force_reinit=True)
        results = retriever.retrieve("What subscription plans are available?", k=3)
        assert len(results) > 0, "No documents retrieved for subscription query"
        top_doc = results[0]
        assert "subscription" in top_doc.document_text.lower() or "plan" in top_doc.document_text.lower()
        assert top_doc.similarity_score > 0.2
        print(f"PASS: Subscription query retrieval (Top: '{top_doc.title} - {top_doc.section}', Score: {top_doc.similarity_score:.4f})")


def test_cancellation_query_retrieval():
    """Test query: 'How do I cancel my subscription?'"""
    with patch.dict(os.environ, {"EMBEDDING_PROVIDER": "local"}):
        retriever = get_retriever(force_reinit=True)
        results = retriever.retrieve("How do I cancel my subscription?", k=3)
        assert len(results) > 0, "No documents retrieved for cancellation query"
        top_doc = results[0]
        assert "cancel" in top_doc.document_text.lower()
        assert top_doc.similarity_score > 0.2
        print(f"PASS: Cancellation query retrieval (Top: '{top_doc.title} - {top_doc.section}', Score: {top_doc.similarity_score:.4f})")


def test_contact_query_retrieval():
    """Test query: 'How can I contact support?'"""
    with patch.dict(os.environ, {"EMBEDDING_PROVIDER": "local"}):
        retriever = get_retriever(force_reinit=True)
        results = retriever.retrieve("How can I contact support?", k=3)
        assert len(results) > 0, "No documents retrieved for contact query"
        top_doc = results[0]
        assert "contact" in top_doc.document_text.lower() or "support" in top_doc.title.lower()
        assert top_doc.similarity_score > 0.2
        print(f"PASS: Contact support retrieval (Top: '{top_doc.title} - {top_doc.section}', Score: {top_doc.similarity_score:.4f})")


def test_openai_provider_selection_mock():
    """Verify that EMBEDDING_PROVIDER=openai selects OpenAIEmbeddingProvider without real API call."""
    with patch.dict(os.environ, {"EMBEDDING_PROVIDER": "openai", "OPENAI_API_KEY": "test-key-mock"}):
        with patch("openai.OpenAI"):
            provider = get_embedding_provider()
            assert isinstance(provider, OpenAIEmbeddingProvider)
    print("PASS: OpenAI embedding provider selection (mocked)")


def test_invalid_provider_selection_error():
    """Verify that EMBEDDING_PROVIDER=invalid raises ValueError."""
    with patch.dict(os.environ, {"EMBEDDING_PROVIDER": "invalid"}):
        try:
            get_embedding_provider()
            assert False, "Expected ValueError for invalid provider"
        except ValueError as e:
            assert "Invalid EMBEDDING_PROVIDER" in str(e)
    print("PASS: Invalid provider error handling")


def run_all_rag_tests():
    print("========================================")
    print("RUNNING DAY 2 KNOWLEDGE BASE & RAG TESTS")
    print("========================================")
    test_document_loading()
    test_document_chunking()
    test_local_provider_selection()
    test_deterministic_embedding_reproducibility()
    test_different_text_produces_different_vectors()
    test_vector_normalization()
    test_refund_policy_retrieval()
    test_shipping_query_retrieval()
    test_subscription_query_retrieval()
    test_cancellation_query_retrieval()
    test_contact_query_retrieval()
    test_openai_provider_selection_mock()
    test_invalid_provider_selection_error()
    print("========================================")
    print("ALL DAY 2 RAG TESTS PASSED SUCCESSFULLY!")
    print("========================================\n")


if __name__ == "__main__":
    run_all_rag_tests()
