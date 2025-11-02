# app/agents/extraction.py  — robust OCR + parsing
import os
import io
import re
import tempfile
from typing import Optional, List, Dict
from dataclasses import dataclass

from PIL import Image, ImageOps, ImageFilter

# Try to import pytesseract and wire the tesseract binary if present
try:
    import pytesseract
    HAS_PYTESS = True
except Exception:
    pytesseract = None  # type: ignore
    HAS_PYTESS = False


def _to_float(val: Optional[str]) -> Optional[float]:
    if val is None:
        return None
    s = str(val).strip().replace(",", ".")
    try:
        return float(s)
    except Exception:
        return None


@dataclass
class ParsedResult:
    test_name: str
    value: Optional[float]
    unit: Optional[str]
    reference_range: Optional[str]
    interpretation: Optional[str]
    raw: str


class DigitalDataExtractionAgent:
    """
    OCR + Lab text parsing.
    - OCR: Pillow-only preprocessing; pytesseract backend.
    - Parsing: SPECIALS w/ named groups + generic LAB_LINE handling hyphens, units, ranges, flags.
    """
    name = "DigitalDataExtractionAgent"

    # Generic analyzer line (non-greedy name, optional separators, units, ref-range, flags)
    LAB_LINE = re.compile(
        r"(?P<name>[A-Za-z][A-Za-z0-9 \-/()+%αβµμ]*?)\s*"
        r"(?:[:\-–—]\s*)?"
        r"(?P<val>-?\d+(?:[.,]\d+)?)\s*"
        r"(?P<unit>[A-Za-z/%μµIU·]+)?"
        r"(?:\s*\((?P<ref>[^)]+)\))?"
        r"(?:\s*(?P<intp>high|low|normal|abnormal|positive|negative))?",
        re.IGNORECASE,
    )

    # Common fertility assays
    SPECIALS = {
        "AMH": re.compile(r"(?i)\bAMH\b.*?(?P<val>\d+(?:[.,]\d+)?)\s*(?P<unit>ng/ml|pmol/l)\b"),
        "FSH": re.compile(r"(?i)\bFSH\b.*?(?P<val>\d+(?:[.,]\d+)?)\s*(?P<unit>IU/L|mIU/mL)\b"),
        "LH": re.compile(r"(?i)\bLH\b.*?(?P<val>\d+(?:[.,]\d+)?)\s*(?P<unit>IU/L|mIU/mL)\b"),
        "Estradiol": re.compile(r"(?i)\b(?:E2|Estradiol)\b.*?(?P<val>\d+(?:[.,]\d+)?)\s*(?P<unit>pg/mL|pmol/L)\b"),
        "Progesterone": re.compile(r"(?i)\bProgesterone\b.*?(?P<val>\d+(?:[.,]\d+)?)\s*(?P<unit>ng/mL|nmol/L)\b"),
        "hCG": re.compile(r"(?i)\b(?:hcg|β-hcg|beta-hcg)\b.*?(?P<val>\d+(?:[.,]\d+)?)\s*(?P<unit>IU/L|mIU/mL)\b"),
    }

    def __init__(self) -> None:
        self._wire_tesseract()

    # ---------- Tesseract wiring ----------
    def _wire_tesseract(self) -> None:
        """Set pytesseract.tesseract_cmd if we can find it. Windows needs this."""
        if not HAS_PYTESS:
            return
        cmd = os.getenv("TESSERACT_CMD")
        if cmd and os.path.isfile(cmd):
            pytesseract.pytesseract.tesseract_cmd = cmd
            return
        # Common Windows install path
        win_path = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
        if os.name == "nt" and os.path.isfile(win_path):
            pytesseract.pytesseract.tesseract_cmd = win_path

    def ocr_ready(self) -> bool:
        """True if pytesseract is importable and the binary appears callable."""
        if not HAS_PYTESS:
            return False
        try:
            _ = pytesseract.get_tesseract_version()
            return True
        except Exception:
            return False

    def tesseract_version(self) -> str:
        if not HAS_PYTESS:
            return "[pytesseract not installed]"
        try:
            return str(pytesseract.get_tesseract_version())
        except Exception as e:
            return f"[tesseract not found] {e}"

    # ---------- OCR ----------
    def _preprocess_image(self, img: Image.Image) -> Image.Image:
        """
        Lightweight preprocessing that works without OpenCV:
        - convert to L (grayscale)
        - upscale 2x (improves OCR on small text)
        - median filter (denoise)
        - autocontrast
        - binarize (threshold)
        """
        g = img.convert("L")
        # upscale
        w, h = g.size
        g = g.resize((w * 2, h * 2), Image.LANCZOS)
        # denoise slightly
        g = g.filter(ImageFilter.MedianFilter(size=3))
        # autocontrast
        g = ImageOps.autocontrast(g)
        # binarize with Otsu-ish simple heuristic
        hist = g.histogram()
        # Use a simple threshold near the valley between peaks if possible; fallback to 128
        try:
            # crude valley pick
            cumsum = 0
            total = sum(hist)
            target = total * 0.5
            thresh = 128
            for i, count in enumerate(hist):
                cumsum += count
                if cumsum >= target:
                    thresh = i
                    break
            thresh = max(90, min(165, thresh))
        except Exception:
            thresh = 128
        bw = g.point(lambda x: 255 if x > thresh else 0, mode="1")
        return bw.convert("L")

    def _ocr_bytes(self, raw: bytes, lang: str = "eng") -> str:
        if not HAS_PYTESS:
            return "[OCR unavailable: install pytesseract + Tesseract]"
        try:
            img = Image.open(io.BytesIO(raw)).convert("RGB")
            proc = self._preprocess_image(img)
            # Persist to temp PNG to avoid PIL decoder edge cases
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                proc.save(tmp.name)
                tmp_path = tmp.name
            try:
                config = "--oem 3 --psm 6"  # LSTM, assume a block of text
                return pytesseract.image_to_string(Image.open(tmp_path), lang=lang, config=config)
            finally:
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
        except Exception as e:
            return f"[Image OCR error] {e}"

    def _ocr_path(self, path: str, lang: str = "eng") -> str:
        try:
            with open(path, "rb") as fh:
                return self._ocr_bytes(fh.read(), lang=lang)
        except Exception as e:
            return f"[Image OCR error] {e}"

    def ocr_image(self, file_bytes_or_path) -> str:
        """
        Accepts raw bytes or a filesystem path. Returns extracted text.
        Why: Gradio may supply a filepath; our chat handler may pass bytes.
        """
        if isinstance(file_bytes_or_path, (bytes, bytearray)):
            return self._ocr_bytes(bytes(file_bytes_or_path))
        if isinstance(file_bytes_or_path, str):
            return self._ocr_path(file_bytes_or_path)
        if hasattr(file_bytes_or_path, "read"):
            try:
                return self._ocr_bytes(file_bytes_or_path.read())
            finally:
                try:
                    file_bytes_or_path.close()
                except Exception:
                    pass
        return "[OCR error] Unsupported input type"

    # ---------- Parsing ----------
    def parse_text(self, text: str) -> List[ParsedResult]:
        out: List[ParsedResult] = []

        # Known assays first
        for name, pat in self.SPECIALS.items():
            for m in pat.finditer(text):
                out.append(
                    ParsedResult(
                        name,
                        _to_float(m.group("val")),
                        (m.group("unit") or "").strip() or None,
                        None,
                        None,
                        m.group(0),
                    )
                )

        # Generic lines
        for line in text.splitlines():
            m = self.LAB_LINE.search(line)
            if not m:
                continue
            out.append(
                ParsedResult(
                    test_name=m.group("name").strip(),
                    value=_to_float(m.group("val")),
                    unit=(m.group("unit") or "").strip() or None,
                    reference_range=(m.group("ref") or None),
                    interpretation=((m.group("intp") or "").lower() or None),
                    raw=line.strip(),
                )
            )

        # Deduplicate by test name; prefer entries with numeric value
        dedup: Dict[str, ParsedResult] = {}
        for p in out:
            k = p.test_name.upper()
            if k not in dedup or (dedup[k].value is None and p.value is not None):
                dedup[k] = p
        return list(dedup.values())
