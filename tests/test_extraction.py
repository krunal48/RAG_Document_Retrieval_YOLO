from app.agents.extraction import DigitalDataExtractionAgent

def test_parser_specials_named_groups():
    agent = DigitalDataExtractionAgent()
    txt = "AMH: 2.1 ng/mL (normal)\nFSH 8.4 IU/L\nEstradiol 210 pg/mL\nProgesterone 0.9 ng/mL\nhCG 5 mIU/mL"
    parsed = agent.parse_text(txt)
    names = {p.test_name.upper() for p in parsed}
    assert {"AMH","FSH","ESTRADIOL","PROGESTERONE","HCG"}.issubset(names)
    amh = [p for p in parsed if p.test_name.upper()=="AMH"][0]
    assert amh.value == 2.1 and amh.unit.lower().startswith("ng/")

def test_parser_generic_line():
    agent = DigitalDataExtractionAgent()
    txt = "RandomAnalyte - 12.5 mg/dL (5-15) high"
    parsed = agent.parse_text(txt)
    assert any(p.test_name.lower().startswith("random") and p.value == 12.5 for p in parsed)
