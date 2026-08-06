import pytest
from utils import (
    EnterpriseEmbeddingEngine,
    QdrantPersistentVectorStore,
    BM25SearchEngine,
    FlashRankReranker,
    HybridRetriever
)

def test_enterprise_embedding_engine():
    engine = EnterpriseEmbeddingEngine()
    embeddings = engine.encode(["Interface GigabitEthernet0/1 down"])
    assert len(embeddings) == 1
    assert len(embeddings[0]) in [1024, 768, 384]

def test_qdrant_vector_store():
    qdrant = QdrantPersistentVectorStore()
    sample_chunk = {
        "content": "%ETHPORT-5-IF_DOWN: Interface GigabitEthernet0/1 is down",
        "filename": "syslog_qdrant_unique.log",
        "title": "syslog_qdrant_unique.log",
        "page": 1,
        "section": "Telemetry",
        "chunk_id": "syslog_q1",
        "document_type": "Server Log / Alert",
        "source_type": "Server Log / Alert",
        "device_type": "Syslog",
        "citation": "[syslog_qdrant_unique.log | Page #1 | Chunk #1]"
    }
    qdrant.upsert_chunks([sample_chunk])
    results = qdrant.search("GigabitEthernet0/1 down", top_k=2)
    assert isinstance(results, list)
    assert len(results) > 0


def test_bm25_search_engine():
    bm25 = BM25SearchEngine()
    chunks = [
        {"content": "Cisco IOS switch configuration interface GigabitEthernet0/1 uplink port", "citation": "cit_1"},
        {"content": "Linux server auth.log invalid password attempt for root user", "citation": "cit_2"},
        {"content": "Standard operating procedure for router recovery and console cable", "citation": "cit_3"}
    ]
    bm25.build_index(chunks)
    res = bm25.search("GigabitEthernet0/1 uplink", top_k=2)
    assert len(res) > 0
    assert "GigabitEthernet0/1" in res[0]["content"]


def test_hybrid_retriever(vector_store):
    results = vector_store.search("GigabitEthernet0/1 down", top_k=2)
    assert isinstance(results, list)
