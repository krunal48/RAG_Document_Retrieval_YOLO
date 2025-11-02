# Fertility Multi‑Agent System — Sprint 1

End‑to‑end demo: **RAG + Pinecone**, **ASHA router + Gradio UI**, **Embryology secure updates**, **OCR + Lab parser**.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...       # set your key
export PINECONE_API_KEY=pc-...     # set your key
# optional:
export PINECONE_INDEX_NAME=fertility-rag
export PINECONE_NAMESPACE=default

# run tests
pytest -q

# run UI
python app/ui/app.py
```
