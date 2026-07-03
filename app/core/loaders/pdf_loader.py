from typing import Tuple, Dict, Any
import pymupdf
from io import BytesIO
from app.core.documents.elements import (
    DocumentElement,
    ElementType,
    ExtractedDocument,
)
import logging
from app.core.loaders.base import DocumentLoader


logger = logging.getLogger(__name__)

_MAX_HEADING_LEVELS = 6
_HEADING_MIN_RATIO = 1.15
_HEADING_RATIO = 1.15       # font size vs page median (Possibly heading)
_TITLE_RATIO = 1.6          # same heuristic for title
_BOLD_HEADING_MAX_LEN = 120

class PDFLoader(DocumentLoader):
    def load(self, file_bytes: bytes) -> ExtractedDocument:
        elements: list[DocumentElement] = []
        warnings: list[str] = []

        with pymupdf.open(stream=BytesIO(file_bytes), filetype="pdf") as doc:
            doc_metadata: dict[str, Any] = {
                "loader": "pdf",
                "page_count": doc.page_count,
                "title": (doc.metadata or {}).get("title") or None,
            }

            size_to_level, body_median = _build_size_level_map(doc)

            heading_stack: list[str] = []

            for page_index in range(doc.page_count):
                try:
                    page = doc.load_page(page_index)
                except Exception as ex:
                    warnings.append(f"page_{page_index+1}_load_failed: {ex}")
                    logger.warning(
                        "pdf_page_load_failed",
                        extra={"page": page_index + 1, "error": str(ex)},
                    )
                    continue

                table_bboxes = _extract_tables_into(
                    page=page,
                    page_index=page_index,
                    heading_path=tuple(heading_stack),
                    out_elements=elements,
                    warnings=warnings,
                )

                try:
                    blocks = page.get_text("dict")["blocks"]
                except Exception as ex:
                    warnings.append(f"page_{page_index+1}_text_extract_failed: {ex}")
                    logger.warning(
                        "pdf_page_text_extraction_failed",
                        extra={"page": page_index + 1, "error": str(ex)},
                    )
                    continue

                for block in blocks:
                    if block.get("type") != 0:  # 0 => text-block, 1 => image
                        continue

                    for line in block.get("lines", []):
                        spans = line.get("spans", [])
                        if not spans:
                            continue

                        line_text = "".join(s["text"] for s in spans).strip()
                        if not line_text:
                            continue
                            
                        line_bbox = line.get("bbox")
                        if line_bbox and _inside_any(line_bbox, table_bboxes):
                            continue

                        # A line's "size" = max span size (the largest span dominates the intent).
                        line_size = max(s["size"] for s in spans)
                        is_bold = any("Bold" in s.get("font", "") for s in spans)

                        level = _classify_level(line_size, is_bold, line_text, size_to_level, body_median)

                        if level is not None:
                            heading_stack = _push_heading(heading_stack, line_text, level)
                            etype = ElementType.TITLE if level == 1 else ElementType.HEADING
                        else:
                            etype = _classify_body(line_text)

                        elements.append(
                            DocumentElement(
                                type=etype,
                                text=line_text,
                                heading_path=tuple(heading_stack),
                                metadata={
                                    "page": page_index + 1,
                                    "font_size": round(line_size, 2),
                                    "bold": is_bold,
                                    "heading_level": level,
                                },
                            )
                        )

            doc_metadata["warnings"] = warnings
            doc_metadata["element_count"] = len(elements)

        return ExtractedDocument(elements=elements, metadata=doc_metadata)


def _extract_tables_into(
    *,
    page,
    page_index: int,
    heading_path: tuple[str, ...],
    out_elements: list[DocumentElement],
    warnings: list[str],
) -> list[tuple[float, float, float, float]]:
    """
    Find tables on the page, render each as markdown, append them to
    `out_elements`, and return their bounding boxes so text extraction can
    skip cell content later.
    """
    try:
        tf = page.find_tables()
    except Exception as exc:
        warnings.append(f"page_{page_index+1}_find_tables_failed: {exc}")
        return []

    bboxes: list[tuple[float, float, float, float]] = []
    for t in tf.tables:
        try:
            rows = t.extract()  # list[list[str | None]]
            md = _rows_to_markdown(rows)
            if not md:
                continue
            out_elements.append(
                DocumentElement(
                    type=ElementType.TABLE,
                    text=md,
                    heading_path=heading_path,
                    metadata={
                        "page": page_index + 1,
                        "rows": len(rows),
                        "cols": len(rows[0]) if rows else 0,
                    },
                )
            )
            bboxes.append(tuple(t.bbox))  # (x0, y0, x1, y1)
        except Exception as exc:
            warnings.append(f"page_{page_index+1}_table_extract_failed: {exc}")

    return bboxes


def _rows_to_markdown(rows: list[list[str | None]]) -> str:
    cleaned = [
        [(c or "").strip().replace("\n", " ") for c in row]
        for row in rows
        if row
    ]
    if not cleaned:
        return ""
    header, *body = cleaned
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for r in body:
        # Pad short rows so markdown stays valid
        if len(r) < len(header):
            r = r + [""] * (len(header) - len(r))
        lines.append("| " + " | ".join(r[: len(header)]) + " |")
    return "\n".join(lines)


def _inside_any(bbox: tuple[float, float, float, float],
                table_bboxes: list[tuple[float, float, float, float]]) -> bool:
    x0, y0, x1, y1 = bbox
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    for tx0, ty0, tx1, ty1 in table_bboxes:
        if tx0 <= cx <= tx1 and ty0 <= cy <= ty1:
            return True
    return False


def _classify_level(
    line_size: float,
    is_bold: bool,
    text: str,
    size_to_level: dict[float, int],
    body_median: float,
) -> int | None:
    """
    Return the heading level (1..6) for this line, or None if it's body text.
    """
    key = round(line_size * 2) / 2
    if key in size_to_level:
        return size_to_level[key]
    # Fallback: bolded short line at body size → treat as deepest heading level.
    if is_bold and len(text) <= _BOLD_HEADING_MAX_LEN and line_size >= body_median:
        # Use the last known heading level if any exist, else level 3 as a sane default.
        return max(size_to_level.values(), default=3) if size_to_level else 3
    return None


def _classify_body(text: str) -> ElementType:
    if text[:2] in ("• ", "- ", "* "):
        return ElementType.LIST_ITEM
    if len(text) > 2 and text[0].isdigit() and text[1] in ".)":
        return ElementType.LIST_ITEM
    return ElementType.PARAGRAPH


def _push_heading(stack: list[str], new_heading: str, level: int) -> list[str]:
    """
    Level-aware push: Truncate the stack to length `level - 1` (pop everything at or below
        the incoming level).
      - Append the new heading.
    """
    stack = stack[: level - 1]
    if len(stack) < level - 1:
        stack = stack + [""] * (level - 1 - len(stack))
    stack.append(new_heading)
    return stack

def _build_size_level_map(doc) -> tuple[dict[float, int], float]:
    """
    Walk the whole document once, collect the distribution of font sizes,
    and assign the top-K distinct sizes to heading levels 1..K.

    Returns:
      size_to_level: mapping from *rounded* font size (0.5pt bucket) to level 1..6
      body_median:   the median font size across body text (used as a sanity floor)
    """
    from collections import Counter
    size_counts: Counter[float] = Counter()

    for page in doc:
        try:
            for block in page.get_text("dict")["blocks"]:
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        # bucket to 0.5pt so tiny anti-aliasing noise doesn't
                        # create fake distinct sizes
                        size_counts[round(span["size"] * 2) / 2] += 1
        except Exception:
            continue

    if not size_counts:
        return {}, 12.0

    # Body text = the size with the most characters. Everything larger and
    # meaningfully-different is a heading candidate.
    body_median = max(size_counts.items(), key=lambda kv: kv[1])[0]

    heading_sizes = sorted(
        (s for s in size_counts if s > body_median * _HEADING_MIN_RATIO),
        reverse=True,
    )
    # Cap at H1..H6
    heading_sizes = heading_sizes[:_MAX_HEADING_LEVELS]

    size_to_level = {size: level for level, size in enumerate(heading_sizes, start=1)}
    return size_to_level, body_median