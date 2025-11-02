import os
import pytest
from app.vector_store.pinecone_adapter import PineconeVectorStore
from app.agents.rag import RAGAgent

PINECONE_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

@pytest.mark.skipif(not (PINECONE_KEY and OPENAI_KEY), reason="Requires Pinecone and OpenAI keys")
def test_rag_ingest_and_search():
    store = PineconeVectorStore()
    rag = RAGAgent(store)
    rag.add_documents([("Clinic IVF Policy", "Day 3 updates include cell counts and fragmentation. Day 5 blastocyst grading follows Gardner.")])
    res = rag.answer("How are Day 5 blastocysts graded?")
    assert "blastocyst" in res.reply.lower()
