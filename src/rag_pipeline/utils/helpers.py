import json
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Union, Optional

__all__ = [
    "_load_json",
    "_is_v1",
    "_is_v2",
    "_v2_to_v1",
    "normalize_parsed_dir_to_v1",
    "try_coerce_number",
    "fix_numbers_in_answers",
]

#  Normalization (V2 -> V1)

def _load_json(p: Path) -> Dict[str, Any]:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def _is_v1(doc: dict) -> bool:
    return isinstance(doc, dict) and "metainfo" in doc and "content" in doc and isinstance(doc.get("content"), list)

def _is_v2(doc: dict) -> bool:
    return isinstance(doc, dict) and "doc_name" in doc and "content" in doc and isinstance(doc.get("content"), list)

def _v2_to_v1(doc: dict) -> dict:
    pages: List[Dict[str, Any]] = []
    for p in doc.get("content", []):
        page_no = p.get("page_number") or p.get("page_id")
        txt = p.get("text") or ""
        if page_no is None:
            continue
        pages.append({
            "page": int(page_no),
            "content": [{"type": "text", "text": txt}],
            "page_dimensions": {},
        })
    pages.sort(key=lambda x: x["page"])
    return {
        "metainfo": {
            "sha1_name": Path(doc.get("doc_name") or "document").stem,
            "pages_amount": len(pages),
            "source_path": doc.get("source_path"),
            "backend": doc.get("backend"),
            "ocr_model": doc.get("ocr_model"),
        },
        "content": pages,
    }

def normalize_parsed_dir_to_v1(input_dir: Path, output_dir: Path) -> int:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for src in input_dir.glob("*.json"):
        try:
            doc = _load_json(src)
            dst = output_dir / src.name
            if _is_v1(doc):
                shutil.copy2(src, dst)
                count += 1
            elif _is_v2(doc):
                v1 = _v2_to_v1(doc)
                with dst.open("w", encoding="utf-8") as f:
                    json.dump(v1, f, ensure_ascii=False, indent=2)
                count += 1
            else:
                print(f"[normalize] skip (unknown schema): {src}")
        except Exception as e:
            print(f"[normalize] error {src}: {e}")
    return count

#  Number coercion

_NUM_SIMPLE_RE = re.compile(r'^[+\-]?\d+(?:[.,]\d+)?$')

def try_coerce_number(val: Any) -> Any:
    if isinstance(val, (int, float)):
        return val
    if not isinstance(val, str):
        return val
    s = val.strip()
    if not s:
        return val
    if s.endswith('%'):
        s = s[:-1].strip()
    s_norm = (
        s.replace('\u00A0', '')
         .replace('\u202F', '')
         .replace(' ', '')
         .replace('_', '')
         .replace(',', '.')
    )
    if re.search(r'[A-Za-zА-Яа-яЁё]', s_norm):
        return val
    if not _NUM_SIMPLE_RE.match(s_norm):
        return val
    try:
        num = float(s_norm)
        return int(num) if num.is_integer() else num
    except Exception:
        return val

def fix_numbers_in_answers(
    input_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None
) -> Path:
    inp = Path(input_path)
    out = Path(output_path) if output_path else inp
    data = json.loads(inp.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Expected a JSON array.")
    for row in data:
        if isinstance(row, dict) and "answer" in row:
            row["answer"] = try_coerce_number(row["answer"])
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote fixed JSON to: {out}")
    return out



def fix_numbers_in_answers_local(
    input_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None
) -> Path:
    inp = Path(input_path)
    out = Path(output_path) if output_path else inp
    data = json.loads(inp.read_text(encoding='utf-8'))
    if not isinstance(data, list):
        raise ValueError("Expected a JSON array at the root.")
    for row in data:
        if isinstance(row, dict) and "answer" in row:
            row["answer"] = try_coerce_number(row["answer"])
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"Wrote fixed JSON to: {out}")
    return out
