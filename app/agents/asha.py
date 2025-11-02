from app.agents.rag import AgentRequest, AgentResponse

class ASHAAgent:
    name = "ASHAAgent"
    def __init__(self, rag, emb, ext):
        self.rag, self.emb, self.ext = rag, emb, ext

    def _intent(self, t: str) -> str:
        t = t.lower()
        if any(k in t for k in ["embryo","embryology","blastocyst","2pn","day 3","day3","day 5","result"]): return "embryology"
        if any(k in t for k in ["upload report","parse report","extract","lab report","medical report"]): return "extraction"
        return "rag"

    def handle(self, req: AgentRequest) -> AgentResponse:
        it = self._intent(req.text)
        if it == "embryology":
            if req.patient_id:
                tmp = self.emb.today(req.patient_id)
                return AgentResponse(tmp["reply"], [], self.name)
            return self.rag.answer(req.text)
        if it == "extraction":
            parsed = self.ext.parse_text(req.text)
            import json
            return AgentResponse(json.dumps([p.__dict__ for p in parsed], indent=2), [], self.name)
        return self.rag.answer(req.text)
