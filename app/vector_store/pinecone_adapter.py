import uuid
from typing import List, Dict, Any, Tuple
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from app.settings import SETTINGS

class PineconeVectorStore:
    def __init__(self):
        assert SETTINGS.pinecone_api_key, "PINECONE_API_KEY missing"
        self.pc = Pinecone(api_key=SETTINGS.pinecone_api_key)
        names = {x["name"] for x in self.pc.list_indexes()}
        if SETTINGS.pinecone_index not in names:
            self.pc.create_index(
                name=SETTINGS.pinecone_index,
                dimension=SETTINGS.embed_dim,
                metric="cosine",
                spec=ServerlessSpec(cloud=SETTINGS.pinecone_cloud, region=SETTINGS.pinecone_region),
            )
        self.index = self.pc.Index(SETTINGS.pinecone_index)
        self.ns = SETTINGS.pinecone_namespace
        self.client = OpenAI()

    def _embed(self, texts: List[str]) -> List[List[float]]:
        em = self.client.embeddings.create(model=SETTINGS.embed_model, input=texts)
        return [d.embedding for d in em.data]

    def add_many(self, items: List[Tuple[str, str]]) -> List[str]:
        vecs = self._embed([t[:SETTINGS.rag_ctx_chars] for _, t in items])
        ids, upserts = [], []
        for (title, text), vec in zip(items, vecs):
            vid = str(uuid.uuid4())
            ids.append(vid)
            upserts.append({"id": vid, "values": vec, "metadata": {"title": title, "text": text}})
        self.index.upsert(vectors=upserts, namespace=self.ns)
        return ids

    def search(self, query: str, k: int) -> List[Dict[str, Any]]:
        qv = self._embed([query])[0]
        res = self.index.query(vector=qv, top_k=k, include_metadata=True, namespace=self.ns)
        out = []
        for i, m in enumerate(res.get("matches", []), start=1):
            md = m.get("metadata", {}) or {}
            out.append({"rank": i, "id": m["id"], "title": md.get("title", f"doc-{m['id']}"),
                        "score": float(m.get("score", 0.0)), "snippet": md.get("text", "")[:800]})
        return out
