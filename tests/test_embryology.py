from app.agents.embryology import EmbryologyStore, EmbryologyResultAgent, now_utc

def test_sign_verify_and_view():
    store = EmbryologyStore()
    agent = EmbryologyResultAgent(store)
    pid = "P123"
    date = now_utc().date().isoformat()
    store.put(pid, date, "Summary", {"2PN": 3, "Day3": 2})
    token = agent.post_update(pid, date, "Summary", {"2PN": 3, "Day3": 2})
    rec = agent.view_by_token(token)
    assert rec["patient_id"] == pid and "visual" in rec
