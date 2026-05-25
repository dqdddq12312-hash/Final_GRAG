PROMPT_EVIDENCE_BATCH = """You are an expert GRI sustainability-reporting auditor judging whether a sustainability report meets the SHALL requirements of disclosure {disclosure_id} ("{disclosure_name}").

For EACH requirement listed below, decide ONE status:
    - "pass"        — evidence directly supports the requirement.
    - "partial"     — partial / weak evidence.
    - "fail"        — evidence contradicts the requirement.
    - "no_evidence" — no relevant evidence in the chunks.

Use ONLY the EVIDENCE CHUNKS section as your knowledge source. Cite at most 5 chunk_ids that directly support each verdict. Keep `rationale` ≤ 400 characters and grounded in the evidence text.

SPECIAL DECISION PATHS — apply ONLY when the EVIDENCE CHUNKS contain explicit text
supporting the path. Silence is NOT a pass; when in doubt, return "no_evidence" / "standard".

(1) Conditional requirement (marked [CONDITIONAL] below; text starts with "if X, …"):
    - If chunks EXPLICITLY state the IF-condition is NOT met (e.g., "this report has not
      been subject to external assurance"), set status="pass" and
      decision_path="condition_not_applicable", citing the chunk(s) that state the
      negated condition.
    - If chunks state the IF-condition IS met, judge the THEN-clause normally
      (pass / partial / fail / no_evidence) with decision_path="standard".
    - If chunks are SILENT on whether the IF-condition is met or not, return
      status="no_evidence" and decision_path="standard". Do NOT guess.

(2) Reported non-existence — for a descriptive requirement asking to describe an item
    (policy / process / committee / mechanism / practice / procedure):
    - If chunks EXPLICITLY state the item does NOT exist (e.g., "the company does not
      have such a policy" or "no formal grievance mechanism exists"), set status="pass"
      and decision_path="reported_non_existence", citing the explicit statement.
    - This path does NOT apply to quantitative requirements ("report the
      number/percentage of …") — use normal status for those.
    - Silence is NOT reported non-existence; return status="no_evidence".

In all other cases, set decision_path="standard".

{material_topic_block}{anchor_block}
{coverage_hint_block}
REQUIREMENTS (grouped by parent stem; the same stem applies to all leaves under it):
{requirements_block}

EVIDENCE CHUNKS (page-pack scope, sorted by NLI relevance):
{evidence_block}

Output strictly the JSON below — no prose, no markdown fences:
{{
  "disclosure_id": "{disclosure_id}",
  "requirement_verdicts": [
    {{
      "requirement_id": "<id>",
      "status": "pass|partial|fail|no_evidence",
      "decision_path": "standard|condition_not_applicable|reported_non_existence",
      "rationale": "<≤400 chars>",
      "citations": ["<chunk_id>", ...],
      "confidence": 0.0,
      "source": "llm"
    }}, ...
  ]
}}
"""

PROMPT_ANCHOR = """You are an expert GRI auditor judging disclosure 2-2 ("Entities included in the organization's sustainability reporting").

You must produce TWO things in a single JSON object:
  (a) `disclosure_batch_verdict` — a verdict per shall-requirement of 2-2.
  (b) `scope_anchor` — the cross-disclosure entity scope this report claims.

`scope_anchor.entities` is the list of named entities in the reporting scope.
`consolidation_basis` is a free-text summary (e.g., "equity method for joint
ventures; full consolidation otherwise"). `entities_sources` cites chunk_ids
backing the entity list. `differs_from_financial` is true iff the report
explicitly notes a difference between sustainability scope and financial
consolidation scope.

{coverage_hint_block}
REQUIREMENTS (2-2):
{requirements_block}

EVIDENCE CHUNKS (sorted by NLI relevance):
{evidence_block}

Output strictly the JSON below — no prose, no markdown fences:
{{
  "disclosure_batch_verdict": {{
    "disclosure_id": "2-2",
    "requirement_verdicts": [
      {{ "requirement_id": "<id>", "status": "pass|partial|fail|no_evidence",
         "rationale": "<≤400 chars>", "citations": ["<chunk_id>", ...],
         "confidence": 0.0, "source": "llm" }}, ...
    ]
  }},
  "scope_anchor": {{
    "entities": ["<name>", ...],
    "consolidation_basis": "<free text>",
    "entities_sources": ["<chunk_id>", ...],
    "differs_from_financial": false
  }}
}}
"""

PROMPT_MATCHER = """You are mapping a reported MATERIAL TOPIC to a base topic from the GRI 11 (Oil and Gas Sector 2021) catalogue.

REPORTED MATERIAL TOPIC: "{material_topic}"

GRI 11 BASE TOPICS (loop A only):
{catalogue_block}

If the reported topic semantically matches one base topic, return its base_topic_id (e.g. "11.1"). Otherwise return "none". ALWAYS include:
  - rationale: ≤ 300 chars explaining the choice in terms of GRI 11 scope (what the base topic covers vs what the reported topic covers).
  - candidates: top-3 base topics you considered with a relative score (0..1). The chosen base_topic_id MUST be candidates[0] when the chosen value is not "none".

Output strictly the JSON below — no prose, no markdown fences:
{{
  "base_topic_id": "<11.X | none>",
  "confidence": 0.0,
  "rationale": "<≤300 chars>",
  "candidates": [
    {{"base_topic_id": "11.X", "score": 0.0}}
  ]
}}
"""

__all__ = ["PROMPT_EVIDENCE_BATCH", "PROMPT_ANCHOR", "PROMPT_MATCHER"]
