import time, json, hmac, hashlib
from typing import Dict, Any, Optional
from dataclasses import dataclass
import datetime as dt
from app.settings import SETTINGS

def now_utc():
    return dt.datetime.now(dt.timezone.utc)

@dataclass
class EmbryologyRecord:
    patient_id: str
    date: str
    summary: str
    counts: Dict[str, int]
    ts: str

class EmbryologyStore:
    def __init__(self):
        self._data: Dict[tuple, EmbryologyRecord] = {}

    def put(self, patient_id: str, date: str, summary: str, counts: Dict[str,int]) -> None:
        self._data[(patient_id, date)] = EmbryologyRecord(patient_id, date, summary, counts or {}, now_utc().isoformat())

    def get(self, patient_id: str, date: str) -> Optional[EmbryologyRecord]:
        return self._data.get((patient_id, date))

class EmbryologyResultAgent:
    name = "EmbryologyResultAgent"
    def __init__(self, store: EmbryologyStore):
        self.store = store

    def _bar(self, n: int) -> str:
        return "█" * min(40, max(1, int(max(0, n))))

    def _visual(self, counts: Dict[str, int]) -> str:
        if not counts: return "(no counts available)"
        return "\n".join([f"{k:12} | {self._bar(v)} ({v})" for k, v in counts.items()])

    def _sign(self, payload: str) -> str:
        return hmac.new(SETTINGS.link_secret.encode(), payload.encode(), hashlib.sha256).hexdigest()

    def post_update(self, pid: str, date: str, summary: str, counts: Dict[str, int]) -> str:
        self.store.put(pid, date, summary, counts)
        exp = int(time.time()) + SETTINGS.link_ttl_seconds
        payload = json.dumps({"pid": pid, "date": date, "exp": exp}, separators=(",", ":"))
        return json.dumps({"p": payload, "s": self._sign(payload)})

    def view_by_token(self, token: str) -> Dict[str, Any]:
        data = json.loads(token); payload, sig = data["p"], data["s"]
        if self._sign(payload) != sig: raise ValueError("bad-sig")
        obj = json.loads(payload)
        if int(obj["exp"]) < int(time.time()): raise ValueError("expired")
        rec = self.store.get(obj["pid"], obj["date"])
        if not rec: raise ValueError("no-update")
        return {"patient_id": rec.patient_id, "date": rec.date, "summary": rec.summary,
                "counts": rec.counts, "visual": self._visual(rec.counts)}

    def today(self, patient_id: str):
        date = now_utc().date().isoformat()
        rec = self.store.get(patient_id, date)
        if not rec:
            return {"reply":"No embryology update for today.","agent":self.name}
        token = self.post_update(patient_id, date, rec.summary, rec.counts)
        reply = f"Embryology update for {date}:\n{rec.summary}\n\nCounts:\n{rec.counts}\n\n{self._visual(rec.counts)}\n\nSecure link: {token}"
        return {"reply":reply,"agent":self.name}
