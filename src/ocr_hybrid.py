
import base64
import io
import logging
from typing import Optional, Tuple

from PIL import Image
import fitz  # PyMuPDF
from openai import OpenAI

log = logging.getLogger("hybrid_ocr")


def render_pdf_page_to_jpeg(pdf_path: str, page_num: int, dpi: int = 300, quality: int = 80) -> bytes:
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    with fitz.open(pdf_path) as doc:
        page = doc.load_page(page_num)
        pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()


def openai_ocr_image_bytes(image_bytes: bytes, model: str = "gpt-4o-mini", timeout: float = 60.0) -> str:
    """
    Vision OCR call. Important: do not pass 'temperature' (some models only accept default=1).
    """
    client = OpenAI()
    b64 = base64.b64encode(image_bytes).decode("ascii")
    data_url = f"data:image/jpeg;base64,{b64}"
    resp = client.chat.completions.create(
        model=model,
        timeout=timeout,
        messages=[
            {"role": "system", "content": "You are an OCR engine. Return plain extracted text only. No comments."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Extract all readable text from this page."},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ],
    )
    return (resp.choices[0].message.content or "").strip()


def combine_docling_and_openai_for_page(
    pdf_path: str,
    page_num: int,
    docling_text: Optional[str],
    docling_confidence: Optional[float] = None,
    min_chars: int = 200,
    min_confidence: float = 0.60,
    openai_model: str = "gpt-4o-mini",
    dpi: int = 300,
    jpeg_quality: int = 80,
) -> Tuple[str, str]:
    dl_text = (docling_text or "").strip()
    needs_fallback = (
        len(dl_text) < min_chars or
        (docling_confidence is not None and docling_confidence < min_confidence)
    )
    if not needs_fallback:
        return dl_text, "docling"

    log.info("OpenAI OCR fallback -> %s page %d", pdf_path, page_num + 1)
    jpg = render_pdf_page_to_jpeg(pdf_path, page_num, dpi=dpi, quality=jpeg_quality)
    ai_text = openai_ocr_image_bytes(jpg, model=openai_model)

    def score(s: str) -> int:
        return sum(c.isprintable() and not c.isspace() for c in s)

    if score(ai_text) >= max(score(dl_text), min_chars):
        log.info("Hybrid choice for %s page %d -> openai", pdf_path, page_num + 1)
        return ai_text, "openai"
    log.info("Hybrid choice for %s page %d -> docling", pdf_path, page_num + 1)
    return dl_text, "docling"

