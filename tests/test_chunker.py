from app.ingest.chunker import chunk_text, build_records

def test_chunk_text_basic():
    text = "A"*2500
    chunks = chunk_text(text, 1000, 200)
    assert len(chunks) == 3
    assert chunks[0].startswith("A")
    assert chunks[-1].endswith("A")

def test_build_records_pages():
    recs = build_records("D1", "T", ["p1 text", "p2 text"], 10, 2)
    assert any(r["metadata"]["page"] == 1 for r in recs)
    assert any(r["metadata"]["page"] == 2 for r in recs)
