
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.rag_pipeline.utils.helpers import fix_numbers_in_answers_local
import os
from dotenv import load_dotenv

from src.rag_pipeline.pipeline import (
    RAGRetriever, RetrieverConfig, RerankConfig,
    save_answers_json,
)

load_dotenv()

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

# ===== paths & constants (no getenv) =====
INPUT_DIR = Path("689af43b20634866094170/Dataset")  # input PDFs
PARSED_DIR = Path("output/parsed_v2_nodocling_ocr_no_tables") # parsed directory, can be empty if SKIP_PARSING=True
PARSED_DIR_NORM = Path(str(PARSED_DIR) + "_v1norm") # normalized-to-V1 for the indexer
INDEX_DIR = Path("output/indices_csAUTO_130_300_no_docling_ocr") # point to an existing index if SKIP_INDEXING=True
QUESTIONS = Path("689af43b20634866094170/questions_public.xlsx") # questions file
ANSWERS_JSON = Path("output/indices_csAUTO_answers.json") # output answers file

MODEL_TEXT = "gpt-5"
MODEL_VISION = "gpt-4o"  # legacy, model removed, for compatibility
JUDGE_MODEL = "gpt-4o"  # legacy, model removed, for compatibility
AUG_MODEL = "gpt-5" # model for question augmentation
RERANK_MODEL = "gpt-5-mini"  # model for reranking

TOP_K = 20
MAX_CTX_CHUNKS = 10             # how many chunks total to consider from top-K documents
MAX_SNIPPET_CHARS = 10000       # max context size for LLM input
DEBUG = True                    # print debug info to console
DEBUG_PRINT_TOP_K = 10        # how many top-K results to print in debug mode
DEBUG_SNIPPET_CHARS = 220      # how many chars of context to print in debug mode

ENABLE_QUESTION_AUG = True # whether to run question augmentation

AUG_TEMPERATURE = 0.0 # temperature for question augmentation

N_WORKERS = 25  # parallel answering workers

# key / path checks
def _ensure_openai_key() -> None:
    if "OPENAI_API_KEY" not in os.environ or not os.environ["OPENAI_API_KEY"]:
        raise SystemExit("Set OPENAI_API_KEY in environment.")

def _require_path(p: Path, name: str) -> None:
    if not p.exists():
        raise SystemExit(f"{name} not found: `{p}`")

# ===== splitting & parallel =====
def _split_dataframe(df, n_parts: int) -> List:
    df = df.reset_index(drop=True).copy()
    df["__orig_pos__"] = range(len(df))
    import numpy as np
    return list(np.array_split(df, n_parts))

def _process_df_chunk(
    df_chunk,
    retriever,
    parsed_for_answers,
    pdf_roots,
    MODEL_TEXT,
    MODEL_VISION,
    JUDGE_MODEL,
    TOP_K,
    MAX_CTX_CHUNKS,
    MAX_SNIPPET_CHARS,
    DEBUG,
    DEBUG_SNIPPET_CHARS,
    DEBUG_PRINT_TOP_K,
):
    from src.rag_pipeline.pipeline import answer_questions_from_xlsx_configurable_models
    answers_chunk = answer_questions_from_xlsx_configurable_models(
        df=df_chunk.drop(columns=["__orig_pos__"]),
        retriever=retriever,
        parsed_dir=parsed_for_answers,
        model_text=MODEL_TEXT,
        model_vision=MODEL_VISION,
        judge_model=JUDGE_MODEL,
        top_k_per_doc=TOP_K,
        max_ctx_chunks=MAX_CTX_CHUNKS,
        max_snippet_chars=MAX_SNIPPET_CHARS,
        pdf_roots=pdf_roots,
        debug=DEBUG,
        debug_snippet_chars=DEBUG_SNIPPET_CHARS,
        debug_print_top_k=DEBUG_PRINT_TOP_K,
    )
    for i, row in enumerate(answers_chunk):
        row["__orig_pos__"] = int(df_chunk.iloc[i]["__orig_pos__"])
    return answers_chunk

def _run_parallel_answers(
    df,
    retriever,
    parsed_for_answers,
    pdf_roots,
    *,
    MODEL_TEXT,
    MODEL_VISION,
    JUDGE_MODEL,
    TOP_K,
    MAX_CTX_CHUNKS,
    MAX_SNIPPET_CHARS,
    DEBUG,
    DEBUG_SNIPPET_CHARS,
    DEBUG_PRINT_TOP_K,
    n_workers: int,
) -> List[Dict[str, Any]]:
    if n_workers <= 1:
        return _process_df_chunk(
            df, retriever, parsed_for_answers, pdf_roots,
            MODEL_TEXT, MODEL_VISION, JUDGE_MODEL,
            TOP_K, MAX_CTX_CHUNKS, MAX_SNIPPET_CHARS,
            DEBUG, DEBUG_SNIPPET_CHARS, DEBUG_PRINT_TOP_K
        )
    parts = _split_dataframe(df, n_workers)
    results: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futs = [
            ex.submit(
                _process_df_chunk,
                part,
                retriever,
                parsed_for_answers,
                pdf_roots,
                MODEL_TEXT,
                MODEL_VISION,
                JUDGE_MODEL,
                TOP_K,
                MAX_CTX_CHUNKS,
                MAX_SNIPPET_CHARS,
                DEBUG,
                DEBUG_SNIPPET_CHARS,
                DEBUG_PRINT_TOP_K,
            )
            for part in parts
        ]
        for f in as_completed(futs):
            results.extend(f.result())
    results.sort(key=lambda r: r.get("__orig_pos__", 0))
    for r in results:
        r.pop("__orig_pos__", None)
    return results

# ===== main flow =====
def main() -> None:
    _ensure_openai_key()
    _require_path(INDEX_DIR, "index_dir")
    _require_path(QUESTIONS, "questions")
    parsed_for_answers = PARSED_DIR_NORM if PARSED_DIR_NORM.exists() else PARSED_DIR
    _require_path(parsed_for_answers, "parsed_dir")

    retriever = RAGRetriever(
        RetrieverConfig(index_dir=INDEX_DIR, unique_by_page=False),
        rerank=RerankConfig(
            enable=True,
            top_k_pre_rerank=100,
            batch_size=100,
            llm_model=RERANK_MODEL,
            weight_faiss=0.15,
            weight_llm=0.85,
        ),
    )

    pdf_roots = [INPUT_DIR, PARSED_DIR, PARSED_DIR_NORM]

    from src.question_augment import (
        GPTQuestionAugmentor,
        augment_questions_xlsx,
        probe_augmentor,
    )

    questions_path = QUESTIONS
    if ENABLE_QUESTION_AUG:
        print("[augment] Running GPT question augmentation...")
        augmentor = GPTQuestionAugmentor(
            model=AUG_MODEL,
            temperature=AUG_TEMPERATURE,
            verbose=True,
        )
        if probe_augmentor(augmentor):
            aug_path = QUESTIONS.parent / f"{QUESTIONS.stem}_aug.xlsx"
            questions_path, changes = augment_questions_xlsx(QUESTIONS, aug_path, augmentor)
            print(f"[augment] Augmented file: {questions_path}")
            if changes:
                print(f"[augment] Sample changes: {changes[:3]}")
        else:
            print("[augment] Probe failed; using original questions.")
    else:
        print("[augment] Disabled.")

    import pandas as pd
    df = pd.read_excel(questions_path)
    print(f"[qa] Questions: {len(df)}")

    print(f"[parallel] Workers: {N_WORKERS}")
    answers = _run_parallel_answers(
        df,
        retriever,
        parsed_for_answers,
        pdf_roots,
        MODEL_TEXT=MODEL_TEXT,
        MODEL_VISION=MODEL_VISION,
        JUDGE_MODEL=JUDGE_MODEL,
        TOP_K=TOP_K,
        MAX_CTX_CHUNKS=MAX_CTX_CHUNKS,
        MAX_SNIPPET_CHARS=MAX_SNIPPET_CHARS,
        DEBUG=DEBUG,
        DEBUG_SNIPPET_CHARS=DEBUG_SNIPPET_CHARS,
        DEBUG_PRINT_TOP_K=DEBUG_PRINT_TOP_K,
        n_workers=N_WORKERS,
    )

    save_answers_json(answers, ANSWERS_JSON)
    fix_numbers_in_answers_local(str(ANSWERS_JSON))
    print(f"[qa] Saved answers -> {ANSWERS_JSON}")

if __name__ == "__main__":
    try:
        t0 = time.perf_counter()
        main()
        print(f"[timing] elapsed {time.perf_counter() - t0:.2f}s")
    except KeyboardInterrupt:
        sys.exit(130)