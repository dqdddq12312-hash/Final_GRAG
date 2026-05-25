import csv as _csv
import json
import logging
from pathlib import Path
from pymilvus import MilvusClient

from . import config

logger = logging.getLogger(__name__)

# Helpers
def _read_json(path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _dump_pydantic(obj):
    """Pydantic model → dict; None / dict pass-through."""
    if obj is None:
        return None
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return obj

# Inputs 
def _query_all_chunks(report_id):
    """Pull all chunks cho 1 report từ Zilliz `report_chunks`."""

    if not config.ZILLIZ_URI:
        raise RuntimeError(
            "ZILLIZ_CLOUD_URI is not set in .env — cannot query chunks"
        )

    client = MilvusClient(uri=config.ZILLIZ_URI, token=config.ZILLIZ_TOKEN)
    output_fields = [
        "chunk_id", "report_id", "page_numbers", "content_text",
        "page_start", "page_end", "section_label", "heading_path",
        "chunk_type", "chunk_index",
    ]

    page_size = 10000
    offset = 0
    chunks = []
    while True:
        try:
            batch = client.query(
                collection_name=config.COLLECTION_REPORT_CHUNKS,
                filter=f'report_id == "{report_id}"',
                output_fields=output_fields,
                limit=page_size,
                offset=offset,
            )
        except Exception as e:
            logger.warning(
                "Milvus query failed (offset=%d): %s — returning %d chunks so far",
                offset, e, len(chunks),
            )
            break

        if not batch:
            break
        chunks.extend(batch)

        if len(batch) < page_size:
            break
        offset += page_size

    # Deserialize list-type fields nếu Milvus trả về dạng JSON string
    for chunk in chunks:
        for list_field in ("page_numbers", "heading_path", "doc_item_labels"):
            v = chunk.get(list_field)
            if isinstance(v, str):
                try:
                    chunk[list_field] = json.loads(v)
                except json.JSONDecodeError:
                    chunk[list_field] = []

    return chunks

def load_report_inputs(report_id):
    """Gom metadata local (sou, gci, non_material) 
    và chunks (Zilliz → fallback file) thành dict inputs cho LangGraph."""
    
    report_dir = Path(config.REPORT_UNITS_DIR) / report_id
    sou = _read_json(report_dir / "sou.json") or {}
    gci = _read_json(report_dir / "gri_content_index.json") or []
    non_material = _read_json(report_dir / "non_material.json") or []

    if not isinstance(gci, list):
        raise ValueError(
            f"gri_content_index.json for {report_id} must be a list, got {type(gci)}"
        )
    if not isinstance(non_material, list):
        raise ValueError(
            f"non_material.json for {report_id} must be a list, got {type(non_material)}"
        )

    all_chunks = _query_all_chunks(report_id)
    if not all_chunks:
        local_chunks_path = report_dir / "report_chunks.json"
        if local_chunks_path.exists():
            logger.info(
                "Zilliz returned 0 chunks for %s — falling back to %s",
                report_id, local_chunks_path,
            )
            local = _read_json(local_chunks_path) or []
            for row in local:
                row.pop("dense_embedding", None)
                row.pop("sparse_embedding", None)
            all_chunks = local

    return {
        "report_id": report_id,
        "sou": sou,
        "gci": gci,
        "non_material": non_material,
        "all_chunks_for_report": all_chunks,
    }

# Outputs 
def write_phase_exports(state):
    """Ghi per-phase JSON export vào `output/<report_id>/`."""
    from . import phase_exports as _pe

    report_id = state["report_id"]
    out_dir = Path(config.OUTPUT_DIR) / report_id
    out_dir.mkdir(parents=True, exist_ok=True)

    phase_results = state.get("phase_results") or {}
    written = {}
    for phase_key, builder in _pe.PHASE_EXPORT_BUILDERS.items():
        if phase_key not in phase_results:
            continue
        filename = config.PHASE_EXPORT_FILENAMES.get(phase_key)
        if not filename:
            continue
        payload = builder(state)
        out_path = out_dir / filename
        out_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        written[phase_key] = str(out_path)
    return written

def write_outputs(state):
    """Ghi compliance_report.json + compliance_summary.csv + per-phase exports."""
    report_id = state["report_id"]
    out_dir = Path(config.REPORT_UNITS_DIR) / report_id
    out_dir.mkdir(parents=True, exist_ok=True)

    phase_results = state.get("phase_results", {})
    final_pr = phase_results.get("final")
    overall_pass = (
        bool(final_pr.artifacts.get("overall_pass", False)) if final_pr else False
    )
    pathway = state.get("claim_pathway", "unclear")

    if pathway == "in_accordance":
        evaluated = ["phase3", "phase4", "phase5", "phase6", "phase7", "phase8"]
    elif pathway == "with_reference":
        evaluated = ["phase1", "phase8_light"]
    else:
        evaluated = []

    report = {
        "output_version": config.OUTPUT_VERSION,
        "report_id": report_id,
        "claim_pathway": pathway,
        "sector_standards": state.get("sector_standards", []),
        "overall": {
            "claim_pathway": pathway,
            "overall_pass": overall_pass,
            "evaluated_phases": {
                p: phase_results[p].status
                for p in evaluated
                if p in phase_results
            },
            "all_phase_statuses": {
                p: pr.status for p, pr in phase_results.items()
            },
        },
        "scope_disclaimer": (
            final_pr.artifacts.get("scope_disclaimer") if final_pr else None
        ),
        "scope_anchor": _dump_pydantic(state.get("scope_anchor")),
        "materiality_map": _dump_pydantic(state.get("materiality_map")),
        "pending_omissions": [
            _dump_pydantic(po) for po in state.get("pending_omissions", [])
        ],
        "phase_results": {p: pr.model_dump() for p, pr in phase_results.items()},
    }

    json_path = out_dir / config.COMPLIANCE_REPORT_FILENAME
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    csv_path = out_dir / config.COMPLIANCE_SUMMARY_FILENAME
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["phase", "status", "n_findings", "notes"])
        for p, pr in phase_results.items():
            # Truncate join finding ở 300 char để giữ context multi-finding
            note = " | ".join(pr.findings)[:300]
            w.writerow([p, pr.status, len(pr.findings), note])
        evaluated_str = " + ".join(evaluated) if evaluated else "(none)"
        w.writerow(
            ["overall", "pass" if overall_pass else "fail", 0,
             f"{pathway} pass requires {evaluated_str} all pass"]
        )

    return {
        "json": str(json_path),
        "csv": str(csv_path),
        "phase_exports": write_phase_exports(state),
    }

__all__ = ["load_report_inputs", "write_outputs"]
