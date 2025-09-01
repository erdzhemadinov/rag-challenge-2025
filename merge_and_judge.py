
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from collections import Counter

logging.basicConfig(level=logging.INFO)

# ================== Config / Constants ==================
EMPTY_STRINGS = {
    "", "n/a", "na", "none", "null", "нет данных", "не указано",
    "not specified", "no data", "-", "—"
}
STRIP_QUOTES_RE = re.compile(r'^[\'"\u00AB\u00BB«»](.*?)[\'"\u00AB\u00BB«»]$')
_NUM_SIMPLE_RE = re.compile(r'^[+\-]?\d+(?:[.,]\d+)?$')
_SPLIT_DELIMS_RE = re.compile(r'[;,/]|(?:\s+\-\s+)|(?:\s+и\s+)|(?:\s+and\s+)', re.IGNORECASE)

PLACEHOLDER_NUMERIC_ZERO_AS_EMPTY = True  # treat pure 0 / 0.0 as empty if needed

APPEND_PROMPT_MERGE_INSTRUCTIONS = """
Инструкция объединения ответов (post-processing):
Если несколько уточняющих ответов:
1. Если все числовые – выбрать максимальное.
2. Если ответы являются списками/перечнями (через запятую, точку с запятой, 'и', '/', '-') – нормализовать элементы (трим, убрать дубликаты), объединить в единый упорядоченный список: сначала по убыванию частоты, затем по первому появлению. Вернуть через ', '.
3. Если один ответ является суперстрокой остальных – вернуть самую информативную (самую длинную).
4. Если есть повторяющиеся формулировки в разных регистрах – унифицировать к первому встретившемуся.
5. Иначе применить голосование по частоте; при ничье – самый длинный непустой.
Не добавлять искусственных данных, только комбинировать присутствующие.
""".strip()


# ================== Utility functions ==================
def is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def is_heuristic_empty(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, str):
        s = v.strip().lower()
        if STRIP_QUOTES_RE.match(s):
            s = STRIP_QUOTES_RE.sub(r'\1', s).strip().lower()
        return s in EMPTY_STRINGS
    if PLACEHOLDER_NUMERIC_ZERO_AS_EMPTY and is_number(v):
        return float(v) == 0.0
    return False


def _try_parse_numeric(s: str) -> Optional[float]:
    s2 = s.strip()
    if not s2:
        return None
    if s2.endswith('%'):
        return None
    core = s2.replace(' ', '').replace(',', '.')
    if _NUM_SIMPLE_RE.match(core):
        try:
            return float(core)
        except ValueError:
            return None
    return None


def normalize_for_vote(val: Any) -> Tuple[str, Any]:
    if val is None:
        return ("__none__", None)
    if is_number(val):
        # Use canonical string of float for comparison while keeping value
        return (f"__num__:{float(val)}", val)
    if isinstance(val, str):
        s = val.strip()
        # remove enclosing quotes/brackets
        m = STRIP_QUOTES_RE.match(s)
        if m:
            s = m.group(1).strip()
        num = _try_parse_numeric(s)
        if num is not None:
            return (f"__num__:{num}", num)
        norm = " ".join(s.lower().split())
        return (norm, s)
    s = str(val).strip()
    num = _try_parse_numeric(s)
    if num is not None:
        return (f"__num__:{num}", num)
    return (" ".join(s.lower().split()), s)


def load_clarifications_collect_all(paths: List[Path]) -> Dict[int, List[Any]]:
    acc: Dict[int, List[Any]] = {}
    for path in paths:
        if not path.exists():
            logging.warning("Clarification file missing: %s", path)
            continue
        with path.open("r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except Exception as e:
                logging.error("Failed to load %s: %s", path, e)
                continue
        if not isinstance(data, list):
            continue
        for rec in data:
            if not isinstance(rec, dict):
                continue
            qid = rec.get("question_id")
            if not isinstance(qid, int):
                continue
            ans = rec.get("answer")
            acc.setdefault(qid, []).append(ans)
    return acc


# ================== Enumeration heuristics ==================
def split_enumeration(s: str) -> List[str]:
    temp = s.replace('\n', ',')
    parts = _SPLIT_DELIMS_RE.split(temp)
    out: List[str] = []
    for part in parts:
        p = part.strip()
        if not p:
            continue
        # strip enclosing quotes
        m = STRIP_QUOTES_RE.match(p)
        if m:
            p = m.group(1).strip()
        if p:
            out.append(p)
    return out


def looks_like_enumeration(s: str) -> bool:
    sl = s.lower()
    if ',' in s or ';' in s:
        return True
    if re.search(r'\bи\b', sl):
        return True
    if '/' in s:
        return True
    if ' - ' in s:
        return True
    return False


def try_combine_lists(candidates: List[str]) -> Optional[Tuple[str, Dict[str, Any]]]:
    enum_like = [c for c in candidates if looks_like_enumeration(c)]
    if len(enum_like) < 2:
        return None
    all_items: List[str] = []
    first_occurrence: Dict[str, int] = {}
    freq: Counter = Counter()
    order_counter = 0
    for c in enum_like:
        items = split_enumeration(c)
        for it in items:
            key = it.lower()
            freq[key] += 1
            if key not in first_occurrence:
                first_occurrence[key] = order_counter
                all_items.append(it)
                order_counter += 1
    if not freq:
        return None
    sorted_items_keys = sorted(freq.keys(), key=lambda k: (-freq[k], first_occurrence[k]))
    # Retrieve original casing from first occurrence
    casing_map = {k.lower(): v for k, v in ((it.lower(), it) for it in all_items)}
    sorted_items = [casing_map[k] for k in sorted_items_keys]
    combined = ", ".join(sorted_items)
    # Ensure it actually adds value (more unique than any single enumeration)
    max_unique_single = 0
    for e in enum_like:
        max_unique_single = max(max_unique_single, len(set(x.lower() for x in split_enumeration(e))))
    if len(sorted_items) > max_unique_single:
        return combined, {
            "combination_used": "list_union",
            "combined_from": enum_like,
            "unique_count": len(sorted_items)
        }
    return None


def collapse_substrings(candidates: List[str]) -> Optional[Tuple[str, Dict[str, Any]]]:
    if not candidates:
        return None
    longest = max(candidates, key=len)
    others = [c for c in candidates if c is not longest]
    if others and all(c.lower() in longest.lower() for c in others):
        return longest, {"combination_used": "substring_collapse", "combined_from": candidates}
    return None


# ================== Type coercion ==================
def coerce_to_original_type(answer: Any, original_type: Optional[type]) -> Any:
    if original_type is None:
        return answer
    if answer is None:
        return None
    # If original was int
    if original_type is int:
        if is_number(answer):
            if float(answer).is_integer():
                return int(answer)
            # fall back: still cast if safe
            return int(round(float(answer)))
        if isinstance(answer, str):
            num = _try_parse_numeric(answer)
            if num is not None and float(num).is_integer():
                return int(num)
    # If original was float
    if original_type is float:
        if is_number(answer):
            return float(answer)
        if isinstance(answer, str):
            num = _try_parse_numeric(answer)
            if num is not None:
                return float(num)
    # If original was str
    if original_type is str:
        if not isinstance(answer, str):
            return str(answer)
    # If original was bool
    if original_type is bool:
        if isinstance(answer, bool):
            return answer
        if isinstance(answer, str):
            sl = answer.strip().lower()
            if sl in {"true", "yes", "да", "1"}:
                return True
            if sl in {"false", "no", "нет", "0"}:
                return False
        if is_number(answer):
            return bool(answer)
    return answer


# ================== Selection logic ==================
def select_best_answer(
    candidates: List[Any],
    combine_lists: bool = True,
    prefer_max_for_numeric: bool = True
) -> Tuple[Any, Dict[str, Any]]:
    info: Dict[str, Any] = {}
    # Filter non-empty heuristic
    non_empty = [c for c in candidates if not is_heuristic_empty(c)]
    if not non_empty:
        info["reason"] = "all_empty"
        return None, info

    # If all numeric (or numeric strings) pick max
    parsed_numbers: List[Tuple[float, Any]] = []
    all_numeric = True
    for c in non_empty:
        if is_number(c):
            parsed_numbers.append((float(c), c))
        elif isinstance(c, str):
            num = _try_parse_numeric(c)
            if num is not None:
                parsed_numbers.append((num, c))
            else:
                all_numeric = False
                break
        else:
            all_numeric = False
            break
    if prefer_max_for_numeric and all_numeric and parsed_numbers:
        max_val = max(parsed_numbers, key=lambda x: x[0])
        info["reason"] = "numeric_max"
        return max_val[1], info

    # Attempt list union
    str_candidates = [str(c) for c in non_empty if isinstance(c, str)]
    if combine_lists and str_candidates:
        combined = try_combine_lists(str_candidates)
        if combined:
            ans, meta = combined
            info.update(meta)
            return ans, info

    # Substring collapse
    if str_candidates:
        collapsed = collapse_substrings(str_candidates)
        if collapsed:
            ans, meta = collapsed
            info.update(meta)
            return ans, info

    # Frequency vote (normalized)
    norm_map: Dict[str, Dict[str, Any]] = {}
    counts: Counter = Counter()
    order: Dict[str, int] = {}
    idx = 0
    for c in non_empty:
        key, converted = normalize_for_vote(c)
        counts[key] += 1
        if key not in order:
            order[key] = idx
            norm_map[key] = {"original": c, "converted": converted}
            idx += 1

    # Highest frequency, tie -> longest string repr
    best_key = sorted(
        counts.keys(),
        key=lambda k: (-counts[k], -len(str(norm_map[k]["original"])), order[k])
    )[0]
    chosen = norm_map[best_key]["original"]
    info["reason"] = "frequency_vote"
    info["frequency"] = counts[best_key]
    return chosen, info


# ================== Merge driver ==================
def merge_answers_preserve_format_json_only(
    original_path: Union[str, Path],
    clarifications_paths: List[Union[str, Path]],
    output_path: Union[str, Path],
    changes_report_path: Optional[Union[str, Path]] = None,
    prefer_max_for_numeric: bool = True,
    combine_lists: bool = True,
    preserve_original_types: bool = True
) -> Dict[str, Any]:
    original_path = Path(original_path)
    clarifications_paths_p = [Path(p) for p in clarifications_paths]

    with original_path.open("r", encoding="utf-8") as f:
        base_data = json.load(f)

    # Build original type map
    original_types: Dict[int, Optional[type]] = {}
    for rec in base_data:
        qid = rec.get("question_id")
        ans = rec.get("answer")
        original_types[qid] = type(ans) if ans is not None else None

    clar_map = load_clarifications_collect_all(clarifications_paths_p)
    changes: Dict[int, Dict[str, Any]] = {}
    merged_count = 0
    changed_count = 0

    for rec in base_data:
        qid = rec.get("question_id")
        orig_answer = rec.get("answer")
        candidates: List[Any] = [orig_answer]
        if qid in clar_map:
            candidates.extend(clar_map[qid])
        # Remove duplicates while preserving order
        seen = set()
        uniq_candidates = []
        for c in candidates:
            marker = json.dumps(c, ensure_ascii=False, sort_keys=True)
            if marker not in seen:
                seen.add(marker)
                uniq_candidates.append(c)
        if len(uniq_candidates) == 1:
            continue  # nothing to merge

        new_answer, meta = select_best_answer(
            uniq_candidates,
            combine_lists=combine_lists,
            prefer_max_for_numeric=prefer_max_for_numeric
        )
        # Type coercion
        orig_type = original_types.get(qid)
        coerced_answer = coerce_to_original_type(new_answer, orig_type) if preserve_original_types else new_answer

        if coerced_answer != orig_answer:
            rec["answer"] = coerced_answer
            changed_count += 1
            change_entry = {
                "original_answer": orig_answer,
                "candidates": uniq_candidates,
                "selected_answer": coerced_answer,
                "selection_meta": meta,
                "original_type": None if orig_type is None else orig_type.__name__,
                "coerced_type": None if coerced_answer is None else type(coerced_answer).__name__
            }
            if preserve_original_types and orig_type and type(coerced_answer) != orig_type:
                change_entry["note"] = "Type differing from original due to unavailable safe coercion"
            changes[qid] = change_entry
        merged_count += 1

    # Save outputs
    with Path(output_path).open("w", encoding="utf-8") as f:
        json.dump(base_data, f, ensure_ascii=False, indent=2)

    if changes_report_path:
        with Path(changes_report_path).open("w", encoding="utf-8") as f:
            json.dump(changes, f, ensure_ascii=False, indent=2)

    summary = {
        "total_questions": len(base_data),
        "questions_with_clarifications": len(clar_map),
        "merged_considered": merged_count,
        "changed": changed_count,
        "changes_report_path": str(changes_report_path) if changes_report_path else None,
        "output_path": str(output_path)
    }
    logging.info("Merge summary: %s", summary)
    return summary


# ================== Public runner (no argparse) ==================
def run_merge(
    original: Union[str, Path],
    clarifications: List[Union[str, Path]],
    out: Union[str, Path],
    changes: Optional[Union[str, Path]] = None,
    prefer_max_for_numeric: bool = True,
    combine_lists: bool = True,
    preserve_original_types: bool = True
) -> Dict[str, Any]:
    return merge_answers_preserve_format_json_only(
        original_path=original,
        clarifications_paths=clarifications,
        output_path=out,
        changes_report_path=changes,
        prefer_max_for_numeric=prefer_max_for_numeric,
        combine_lists=combine_lists,
        preserve_original_types=preserve_original_types
    )


# ================== Example invocation template ==================
if __name__ == "__main__":

    summary = run_merge(
        original="output/indices_csAUTO_answers_2025.json",
        clarifications=["output/indices_csAUTO_answers_2025_v2.json", "output/indices_csAUTO_answers_2025_v3.json",
                        "output/indices_csAUTO_answers_2025_v4.json"],
        out="output/indices_csAUTO_answers_2025_merged_last.json",
        changes="output/indices_csAUTO_answers_2025_changes_last.json"
    )
    print(summary)
