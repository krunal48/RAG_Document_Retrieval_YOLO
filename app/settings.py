import os
from dataclasses import dataclass

@dataclass(frozen=True)
class Settings:
    chat_model: str = os.getenv("CHAT_MODEL", "gpt-5-nano")
    embed_model: str = os.getenv("EMBED_MODEL", "text-embedding-3-small")
    embed_dim: int = int(os.getenv("EMBED_DIM", "1536"))
    pinecone_api_key: str = os.getenv("PINECONE_API_KEY", "pcsk_2PeRrb_HGEk5wHasLk6s5jZ3ByuLHgcEQZmbQjAewvBy6jJToFmZhRYnHKunMEvW389zQF")
    pinecone_index: str = os.getenv("PINECONE_INDEX_NAME", "fertility-rag")
    pinecone_namespace: str = os.getenv("PINECONE_NAMESPACE", "default")
    pinecone_cloud: str = os.getenv("PINECONE_CLOUD", "aws")
    pinecone_region: str = os.getenv("PINECONE_REGION", "us-east-1")
    rag_top_k: int = int(os.getenv("RAG_TOP_K", "4"))
    rag_ctx_chars: int = int(os.getenv("RAG_CTX_CHARS", "7000"))
    link_secret: str = os.getenv("LINK_SECRET", "change-me")
    link_ttl_seconds: int = int(os.getenv("LINK_TTL_SECONDS", "3600"))
    system_prompt: str = "You are a careful fertility clinic assistant. \n" \
                         "Answer strictly from the provided context; if missing, say so. \n" \
                         "Avoid diagnosis; recommend contacting clinic staff for personalized advice. \n" \
                         "Cite context blocks as [n]."

SETTINGS = Settings()
