
import os
import sys
from pathlib import Path

from src.rag_pipeline.pipeline import (
    ParseConfig, IndexerConfig, ChunkingConfig, HybridConfig,
    RAGIndexer, parse_documents_pymupdf_openai, build_corpus_and_indices,
    extract_pdf_tables_to_markdown, merge_tables_md_into_parsed,
    parse_workbooks_openpyxl,
)

from src.rag_pipeline.utils.helpers import (
    normalize_parsed_dir_to_v1, fix_numbers_in_answers
)

from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

if not os.environ["OPENAI_API_KEY"]:
    raise SystemExit("Set OPENAI_API_KEY")

# --- toggles ---
SKIP_PARSING = False
SKIP_INDEXING = False
PARSE_XLSX = True
EXPORT_TABLES_MD = False
MERGE_TABLES_INTO_PARSED = False

MIN_CHUNK_TOKENS = 130
MAX_CHUNK_TOKENS = 300
OVERLAP = 0.2

# --- paths
INPUT_DIR = Path("689af43b20634866094170/Dataset")
PARSED_DIR = Path("output/parsed_v2_nodocling_ocr_no_tables")  # may contain mixed V1/V2 parsed JSON
PARSED_DIR_NORM = Path(str(PARSED_DIR) + "_v1norm")  # normalized-to-V1 for the indexer
INDEX_DIR = Path(f"output/indices_csAUTO_{MIN_CHUNK_TOKENS}_{MAX_CHUNK_TOKENS}_no_docling_ocr")
QUESTIONS = Path("689af43b20634866094170/questions_public.xlsx")
TABLES_DIR = Path("output/tables_md")

# --- base configs ---
parse_cfg = ParseConfig(
    input_dir=INPUT_DIR,
    output_dir=PARSED_DIR,
    docling_backend=False,  # if False using PyMuPDF+OpenAI OCR path
)

# --- pipeline ---
def main() -> None:

    if not parse_cfg.docling_backend:
        if not SKIP_PARSING:
            parse_documents_pymupdf_openai(
                input_dir=parse_cfg.input_dir,
                output_dir=parse_cfg.output_dir,
                min_chars_for_ocr=1000,
                image_dpi=200,
                ocr_model="gpt-4o-mini",
                skip_existing = True,
                force_reparse = False
            )
            if PARSE_XLSX:
                parse_workbooks_openpyxl(
                    input_dir=parse_cfg.input_dir,
                    output_dir=parse_cfg.output_dir,
                )
            if EXPORT_TABLES_MD:
                TABLES_DIR.mkdir(parents=True, exist_ok=True)
                extract_pdf_tables_to_markdown(parse_cfg.input_dir, TABLES_DIR)
                if MERGE_TABLES_INTO_PARSED:
                    merged = merge_tables_md_into_parsed(parse_cfg.output_dir, TABLES_DIR)
                    print(f"[tables->parsed] merged into {merged} parsed JSON files")

        # Normalize parsed JSONs to V1 and index
        if not SKIP_INDEXING:
            if not PARSED_DIR.exists():
                raise SystemExit(f"parsed_dir not found: `{PARSED_DIR}`")
            n = normalize_parsed_dir_to_v1(PARSED_DIR, PARSED_DIR_NORM)
            print(f"[normalize] wrote {n} files to `{PARSED_DIR_NORM}`")

            index_cfg = IndexerConfig(
                parsed_dir=PARSED_DIR_NORM,
                index_dir=INDEX_DIR,
                chunking=ChunkingConfig(mode="auto",
                                        min_chunk_tokens=MIN_CHUNK_TOKENS,
                                     max_chunk_tokens=MAX_CHUNK_TOKENS,
                                        overlap_ratio=OVERLAP),
                hybrid=HybridConfig(enable=True, weight_faiss=1.0, weight_lexical=0.0),
            )
            RAGIndexer(index_cfg).build_all()
        else:
            if not INDEX_DIR.exists():
                raise SystemExit(f"index_dir not found: `{INDEX_DIR}`")
    else:
        # Docling path
        index_cfg = IndexerConfig(
            parsed_dir=parse_cfg.output_dir,
            index_dir=INDEX_DIR,
            chunking=ChunkingConfig(mode="auto",
                                    min_chunk_tokens=MIN_CHUNK_TOKENS,
                                    max_chunk_tokens=MAX_CHUNK_TOKENS,
                                    overlap_ratio=OVERLAP),
            hybrid=HybridConfig(enable=True, weight_faiss=1.0, weight_lexical=0.0),
        )
        if not SKIP_PARSING and not SKIP_INDEXING:
            build_corpus_and_indices(parse_cfg, index_cfg)
        elif SKIP_PARSING and not SKIP_INDEXING:
            if not parse_cfg.output_dir.exists():
                raise SystemExit(f"parsed_dir not found: `{parse_cfg.output_dir}`")
            RAGIndexer(index_cfg).build_all()
        else:
            if not INDEX_DIR.exists():
                raise SystemExit(f"index_dir not found: `{INDEX_DIR}`")
    print("Done.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
