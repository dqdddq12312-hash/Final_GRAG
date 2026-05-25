import re
import unicodedata
from collections import Counter, defaultdict

from docling.chunking import HybridChunker
from docling_core.types.doc.document import ContentLayer, DoclingDocument
from docling_core.types.doc.labels import DocItemLabel

HARD_DROP_HEADING_PATTERNS = [
    r"\bgri\s+content\s+index\b",
    r"\bcontent\s+index\b",
    r"\btable\s+of\s+contents?\b",
    r"\babout\s+(this|the)\s+report\s+cover\b",
]

SOFT_DROP_HEADING_PATTERNS = [
    r"\bindependent\s+auditor",
    r"\b(independent\s+)?assurance\s+(statement|report)\b",
    r"\bappendix\b",
]

NOISE_LABELS = frozenset({
    DocItemLabel.PAGE_HEADER.value,
    DocItemLabel.PAGE_FOOTER.value,
    DocItemLabel.FOOTNOTE.value,
    DocItemLabel.DOCUMENT_INDEX.value,
})

# Ký tự zero-width và soft-hyphen
_ZERO_WIDTH_RE = re.compile(r"[\u200b\u200c\u200d\u00ad\ufeff]")
# Ký tự điều khiển không phải newline/tab
_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
# Khoảng trắng liên tiếp
_WHITESPACE_RE = re.compile(r"\s+")
# Hai dấu chấm liên tiếp (lỗi Docling)
_DOUBLE_PERIOD_RE = re.compile(r"\.\s*\.")
# Từ bị ngắt dòng bằng gạch nối (ví dụ: "emis-\nsion" -> "emission")
_HYPHEN_LINEBREAK_RE = re.compile(r"(\w)-\s+([a-z])")
# Ký hiệu hóa học/đơn vị bị tách sai: "tco 2", "co 2", "no x", ...
_CHEM_SPLIT_RE = re.compile(
    r"\b(t?co|n[ho2x]+|so[x2]?|ch4|n2o|nox|sox)\s+(\d)\b",
    re.IGNORECASE,
)
# Ký tự được tính là "số/đặc biệt" để tính tỉ lệ nhiễu
_NUMERIC_CHAR_RE = re.compile(r"[0-9%$.,():\-/ ]")

# Trích xuất metadata từ chunks
def doc_item_labels(chunk):
    """Trả về danh sách label (string) của các doc_item trong chunk."""
    labels = []
    for item in chunk.meta.doc_items:
        if item.label is not None:
            labels.append(str(item.label.value))
    return labels

def chunk_pages(chunk):
    """Trả về danh sách số trang (int, tăng dần) mà chunk xuất hiện."""
    pages = set()
    for item in chunk.meta.doc_items:
        for prov in getattr(item, "prov", []) or []:
            page_no = getattr(prov, "page_no", None)
            if page_no is not None:
                pages.add(int(page_no))
    return sorted(pages)

def heading_path(chunk):
    """Trả về danh sách heading dẫn đến chunk (đã strip khoảng trắng)."""
    headings = getattr(chunk.meta, "headings", None) or []
    return [h.strip() for h in headings if h and h.strip()]

# Text cleaning
def clean_text(text):
    """Làm sạch text: chuẩn hóa Unicode, bỏ ký tự rác, nối từ bị ngắt dòng."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = _ZERO_WIDTH_RE.sub("", text)
    text = _CTRL_RE.sub(" ", text)
    text = _HYPHEN_LINEBREAK_RE.sub(r"\1\2", text)
    text = _CHEM_SPLIT_RE.sub(lambda m: m.group(1) + m.group(2), text)
    text = _DOUBLE_PERIOD_RE.sub(".", text)
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()

def numeric_ratio(text):
    """Tính tỉ lệ ký tự số/đặc biệt trong text (dùng để lọc chunk toàn số)."""
    if not text:
        return 0.0
    matches = _NUMERIC_CHAR_RE.findall(text)
    return len(matches) / max(len(text), 1)

# Main function
def build_clean_chunks(doc, chunker, report_id):
    """Chạy toàn bộ pipeline làm sạch và trả về (cleaned_chunks, cleaning_report)."""
    hard_patterns = [re.compile(p, re.IGNORECASE) for p in HARD_DROP_HEADING_PATTERNS]
    soft_patterns = [re.compile(p, re.IGNORECASE) for p in SOFT_DROP_HEADING_PATTERNS]

    raw_chunks = list(chunker.chunk(doc))

    stage_drops = defaultdict(int)
    samples_dropped = defaultdict(list)

    def record_drop(stage, chunk, reason):
        """Ghi nhận một chunk bị loại: đếm số lượng và lưu tối đa 5 mẫu."""
        stage_drops[stage] += 1
        if len(samples_dropped[stage]) < 5:
            samples_dropped[stage].append({
                "reason": reason,
                "heading_path": heading_path(chunk),
                "doc_item_labels": doc_item_labels(chunk),
                "pages": chunk_pages(chunk),
                "text_preview": (chunk.text or "")[:200],
            })

    #  Bước 1: Loại bỏ FURNITURE và noise-label
    survivors_after_meta = []
    for chunk in raw_chunks:
        # Kiểm tra tất cả doc_items có phải FURNITURE không
        layers = []
        for item in chunk.meta.doc_items:
            layer = getattr(item, "content_layer", None)
            if layer is not None:
                layers.append(str(layer.value))
        if layers and all(layer == ContentLayer.FURNITURE.value for layer in layers):
            record_drop("stage_1_furniture", chunk, "all doc_items in FURNITURE layer")
            continue

        # Kiểm tra tất cả doc_items có phải noise-label không
        labels = doc_item_labels(chunk)
        if labels and all(lbl in NOISE_LABELS for lbl in labels):
            record_drop(
                "stage_1_noise_labels",
                chunk,
                f"all labels in {sorted(NOISE_LABELS)}",
            )
            continue

        survivors_after_meta.append(chunk)

    # Bước 2: lọai bỏ chunks thuộc heading trong danh sách lọc
    survivors_after_heading = []
    for chunk in survivors_after_meta:
        hp = heading_path(chunk)

        # Kiểm tra hard patterns (loại hoàn toàn: GRI content index, TOC, ...)
        if hp:
            joined = " > ".join(hp)
            matched_hard = None
            for pat in hard_patterns:
                if pat.search(joined):
                    matched_hard = pat.pattern
                    break
            if matched_hard:
                record_drop("stage_2_heading_hard", chunk, f"heading matches /{matched_hard}/")
                continue

        # Kiểm tra soft patterns (loại: auditor / assurance / appendix)
        if hp:
            joined = " > ".join(hp)
            matched_soft = None
            for pat in soft_patterns:
                if pat.search(joined):
                    matched_soft = pat.pattern
                    break
            if matched_soft:
                record_drop("stage_2_heading_soft", chunk, f"heading matches /{matched_soft}/")
                continue

        survivors_after_heading.append(chunk)

    # Bước 3: Loại bỏ boilerplate lặp nhiều trang
    # Thu thập các text ngắn và trang chúng xuất hiện
    short_text_pages = defaultdict(set)
    for chunk in survivors_after_heading:
        text_norm = _WHITESPACE_RE.sub(" ", (chunk.text or "").strip()).lower()
        if not text_norm or len(text_norm) > 80:
            continue
        for p in chunk_pages(chunk):
            short_text_pages[text_norm].add(p)

    # Bất kỳ text nào xuất hiện trên >= 5 trang khác nhau → là boilerplate
    boilerplate_set = set()
    for text, pages in short_text_pages.items():
        if len(pages) >= 5:
            boilerplate_set.add(text)

    survivors_after_boilerplate = []
    for chunk in survivors_after_heading:
        text_norm = _WHITESPACE_RE.sub(" ", (chunk.text or "").strip()).lower()
        if text_norm in boilerplate_set:
            record_drop(
                "stage_3_repeated_boilerplate",
                chunk,
                f"appears on {len(short_text_pages[text_norm])} pages",
            )
            continue
        survivors_after_boilerplate.append(chunk)

    # Bước 4 + 5: Clean text và lọc độ dài / tín hiệu
    intermediate = []  # danh sách (chunk, cleaned_text)
    for chunk in survivors_after_boilerplate:
        cleaned = clean_text(chunk.text or "")

        if len(cleaned) < 20:
            record_drop(
                "stage_5_too_short",
                chunk,
                f"char_count={len(cleaned)} < 20",
            )
            continue

        if len(cleaned) < 80 and numeric_ratio(cleaned) > 0.7:
            record_drop(
                "stage_5_numeric_only",
                chunk,
                f"numeric_ratio={numeric_ratio(cleaned):.2f}, len={len(cleaned)}",
            )
            continue

        intermediate.append((chunk, cleaned))

    # Bước 6: Sắp xếp theo trang và build dict chunk
    # Sắp xếp theo (trang bắt đầu, charspan đầu tiên) để giữ thứ tự trong tài liệu
    def sort_key(item):
        chunk = item[0]
        pages = chunk_pages(chunk)
        page_start = pages[0] if pages else 0
        first_charspan = 0
        for d in chunk.meta.doc_items:
            for prov in getattr(d, "prov", []) or []:
                cs = getattr(prov, "charspan", None)
                if cs:
                    first_charspan = int(cs[0])
                    break
            if first_charspan:
                break
        return (page_start, first_charspan)

    intermediate.sort(key=sort_key)

    tokenizer = getattr(chunker, "tokenizer", None)
    cleaned_chunks = []
    for idx, (chunk, cleaned_text) in enumerate(intermediate):
        pages = chunk_pages(chunk)
        hp = heading_path(chunk)
        labels = sorted(set(doc_item_labels(chunk)))

        token_count = 0
        if tokenizer is not None:
            try:
                token_count = int(tokenizer.count_tokens(cleaned_text))
            except Exception:
                token_count = 0

        cleaned_chunks.append({
            "chunk_id": f"{report_id}__c_{idx + 1:04d}",
            "report_id": report_id,
            "chunk_index": idx,
            "modality": "text",
            "chunk_type": "narrative",
            "content_text": cleaned_text,
            "page_numbers": pages,
            "page_start": pages[0] if pages else None,
            "page_end": pages[-1] if pages else None,
            "heading_path": hp,
            "section_label": hp[-1] if hp else "",
            "doc_item_labels": labels,
            "char_count": len(cleaned_text),
            "token_count": token_count,
        })

    cleaning_report = {
        "report_id": report_id,
        "raw_chunk_count": len(raw_chunks),
        "final_chunk_count": len(cleaned_chunks),
        "stage_drops": dict(stage_drops),
        "samples_dropped": {k: v for k, v in samples_dropped.items()},
        "config": {
            "drop_furniture": True,
            "drop_noise_labels": True,
            "hard_drop_heading_patterns": list(HARD_DROP_HEADING_PATTERNS),
            "soft_drop_heading_patterns": list(SOFT_DROP_HEADING_PATTERNS),
            "repeated_boilerplate_min_pages": 5,
            "repeated_boilerplate_max_chars": 80,
            "min_char_count": 20,
            "numeric_ratio_threshold": 0.7,
            "numeric_short_max_chars": 80,
        },
    }

    return cleaned_chunks, cleaning_report
