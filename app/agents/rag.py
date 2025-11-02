from typing import Optional
from dataclasses import dataclass
from openai import OpenAI
from app.settings import SETTINGS

@dataclass
class AgentRequest:
    text: str
    patient_id: Optional[str] = None

@dataclass
class AgentResponse:
    reply: str
    citations: list
    agent: str

class RAGAgent:
    name = "RAGAgent"
    def __init__(self, store):
        self.store = store
        self.client = OpenAI()

    def add_documents(self, items: list) -> list:
        return self.store.add_many(items)

    def answer(self, query: str, k: Optional[int] = None) -> AgentResponse:
        hits = self.store.search(query, k or SETTINGS.rag_top_k)
        if not hits:
            return AgentResponse("No documents in the vector store yet.", [], self.name)
        ctx = "\n\n".join([f"[{h['rank']}] {h['title']}\n{h['snippet']}" for h in hits])[:SETTINGS.rag_ctx_chars]
        prompt = f"Context:\n{ctx}\n\nQuestion: {query}\nAnswer with citations like [1],[2]."
        resp = self.client.chat.completions.create(
            model=SETTINGS.chat_model, temperature=0,
            messages=[{"role":"system","content":SETTINGS.system_prompt},{"role":"user","content":prompt}]
        )
        ans = resp.choices[0].message.content.strip()
        return AgentResponse(ans, [{"rank":h["rank"],"title":h["title"],"score":round(h["score"],4)} for h in hits], self.name)
