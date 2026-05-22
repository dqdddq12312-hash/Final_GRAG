# CHỦ ĐỀ: NGHIÊN CỨU XÂY DỰNG MÔ HÌNH KIỂM TRA TÍNH TUÂN THỦ CỦA BÁO CÁO PHÁT TRIỂN BỀN VỮNG

Khoá luận đề xuất một pipeline kiểm tra tuân thủ tự động cho báo cáo phát triển bền vững theo bộ tiêu chuẩn GRI. Hệ thống nhận đầu vào là PDF báo cáo, trích xuất thành các chunks (có bảo toàn ngữ cảnh) và xử lý làm sạch, sau đó chạy qua một quy trình gồm 9 phase (claim → reporting principles → GRI 2 → material topics → GRI 3 → topic standards → omissions → content index → notify) với một LLM judge (`qwen3:14b` qua Ollama) ở Phase 3/5/6 để cho ra phán quyết `pass / partial / no_evidence / fail` kèm bằng chứng tương cho quyết định.

Module 3 được augment bằng **NLI cross-encoder** (`cross-encoder/nli-deberta-v3-base`) theo hai cơ chế REORDER chunks theo entailment, HINT block trong prompt. Hai thực nghiệm của Chương 4 — TN1 (factorial 2×2 cho REORDER × HINT) và TN2 (xây dựng tập đánh giá ba tầng + calibrate LLM-as-judge arbiter) — đo lường hiệu quả của augmentation và bằng chứng độ tin cậy của tập đánh gái.

## Tổng quan kiến trúc

```
GRI Standards PDF              Báo cáo dầu khí (28 PDF)
      │                                  │
      ▼                                  ▼
  Module 1                           Module 2
  GRI Metadata                       Report Processing
      │                                  │
      └─────────────┬────────────────────┘
                    ▼
              Module 3
        Compliance Engine (LangGraph)
                    │
                    ▼
        compliance_report.json
        compliance_summary.csv
```

## Cấu trúc thư mục

```
Final_GRAG/
├── src/
│   ├── compliance/ 
│   │   ├── graph.py                        # Build graph + routing
│   │   ├── state.py                        # Schema 
│   │   ├── config.py                       # Tham số của hệ thống
│   │   ├── phases/                         # Folder chứa toàn bộ logic kiểm tra tuân thủ Module 3
│   │   └── ... 
│   ├── embedding_utils.py                  # Embedding
│   └── report_chunk_cleaner.py             # Làm sạch dữ liệu báo cáo
│
├── notebook/                 
│   ├── gri_metadata/
│   │   ├── gri_metadata_extraction.ipynb   # Xử lý Tiêu chuẩn GRI
│   │   └── sector_standards.ipynb          # Xử lý Tiêu chuẩn ngành
│   └── report_processing/
│       ├── report_processing.ipynb         # Xử lý Báo cáo phát triển bền vững
│       └── parse_gri_ci.ipynb              # Xử lý GRI content index
│
├── experiments/module3_nli/    
│   ├── nli_lib/                            # Helper library 
│   ├── prompts/                            # 3 phiên bản prompt LLM arbiter (TN2)
│   │
│   ├── run_variant_pipeline.ipynb          # Chạy Module 3 cho 4 cấu hình NLI × 16 báo cáo
│   ├── run_pairing.py                      # Ghép verdict 4 biến thể NLI để so sánh
│   ├── run_human_sampling.py               # Lấy mẫu cho người chấm tay (TN2)
│   ├── run_arbiter.py                      # LLM arbiter chấm các case không đồng thuận (TN2)
│   │
│   ├── tn3_prefix_factorial.ipynb          # TN1 — phân tích pre-fix
│   ├── tn3_postfix_bugfix.ipynb            # TN1 — phân tích post-fix trên ground truth người chấm
│   ├── tn4_groundtruth.ipynb               # TN2 — calibrate độ tin cậy LLM arbiter vs người chấm
│   │
│   ├── data/                               # Dữ liệu của thực nghiệm
│   ├── outputs/                            # Bảng + biểu 
│   └── variant_runs/                       # Kết quả Kiểm tra tính tuân thủ cho 4 cấu hình NLI × 16 báo cáo
│
├── metadata/                               
│   ├── gri_units/                          # Dữ liệu Tiêu chuẩn GRI sau xử lý
│   ├── sector_standards/                   # Dữ liệu Tiêu chuẩn ngành sau xử lý
│   └── report_units/                       # Dữ liệu Báo cáo phát triển bền vững sau xử lý
|                      
├── reports/   
└── GRI_standards/     
```

## Full reproduction

### Setup môi trường

```bash
python -m venv .venv
.venv\Scripts\activate        
pip install -r requirements.txt
```

Phục vụ quá trình chạy:


| Dịch vụ                                                 | Mục đích                                   |
| ------------------------------------------------------- | ------------------------------------------ |
| [Ollama](https://ollama.com/) + `ollama pull qwen3:14b` | LLM judge ở Phase 3/5/6                    |
| [Zilliz Cloud](https://cloud.zilliz.com/)               | Vector DB cho 1 collection `report_chunks` |
| OpenAI API key (`gpt-4o-mini`)                          | LLM arbiter cho TN2                        |
| GPU ≥ 16 GB VRAM                                        | qwen3:14b chế độ thinking + BGE-M3         |


### Thứ tự chạy


| Stage              | Bước                                            | Notebook                                              | Output                                                                              |
| ------------------ | ----------------------------------------------- | ----------------------------------------------------- | ----------------------------------------------------------------------------------- |
| **A. Ingest**      | A.1 Parse GRI standards                         | `notebook/gri_metadata/gri_metadata_extraction.ipynb` | `metadata/gri_units/`                                                               |
|                    | A.2 Parse GRI 11 sector                         | `notebook/gri_metadata/sector_standards.ipynb`        | `metadata/sector_standards/gri_11/`                                                 |
|                    | A.3 Parse GRI Content Index                     | `notebook/report_processing/parse_gri_ci.ipynb`       | `metadata/report_units/<id>/{gri_content_index, sou, non_material}.json`            |
|                    | A.4 Chunk + embed báo cáo → Zilliz              | `notebook/report_processing/report_processing.ipynb`  | `metadata/report_units/<id>/report_chunks.json` + collection `report_chunks`        |
| **B. Pipeline**    | B.1 Run compliance 16 báo cáo × 4 variant       | `experiments/module3_nli/run_variant_pipeline.ipynb`  | `experiments/module3_nli/variant_runs/{a0,a1,a2,v_new}/<id>/compliance_report.json` |
| **C. Thực nghiệm** | C.1 Pair verdict 4 variant                      | `experiments/module3_nli/run_pairing.py`              | `experiments/module3_nli/data/{pairing, verdicts_long, disclosure_index}.csv`       |
|                    | C.2 Stratified disagreement sample (TN2 tầng 1) | `experiments/module3_nli/run_human_sampling.py`       | `experiments/module3_nli/data/sample_{main,agreement,retest}.csv`                   |
|                    | C.3 Run 3 prompt arbiter (TN2 tầng 2)           | `experiments/module3_nli/run_arbiter.py`              | `experiments/module3_nli/data/{arbiter_cache_v*, adjudicated_v*}.{jsonl,csv}`       |
|                    | C.4 TN2 — κ panel, marginal, sign-reversal      | `experiments/module3_nli/tn4_groundtruth.ipynb`       | `experiments/module3_nli/outputs/` (T_TN4_, F5, TN4_.json)                          |
|                    | C.5 TN1 pre-fix — factorial 2×2 hybrid silver   | `experiments/module3_nli/tn3_prefix_factorial.ipynb`  | `experiments/module3_nli/outputs/` (F3, F4, T_TN3pre_)                              |
|                    | C.6 TN1 post-fix — bug-fix lift trên GT v2      | `experiments/module3_nli/tn3_postfix_bugfix.ipynb`    | `experiments/module3_nli/outputs/` (F6–F9, T9–T13)                                  |


## Cấu hình

### `src/compliance/config.py` — các tham số quan trọng

```python
OLLAMA_LLM_MODEL = "qwen3:14b"          # LLM judge
OLLAMA_TEMPERATURE = 0.0
OLLAMA_NUM_CTX = 16384
ENABLE_THINKING_JUDGE = True             # Chế độ thinking — tăng accuracy nhưng chậm

NLI_MODEL = "cross-encoder/nli-deberta-v3-base"
NLI_REORDER_ENABLED = True               # Bật REORDER (override bằng env NLI_REORDER_ENABLED=0)
NLI_HINT_ENABLED = True                  # Bật HINT block (override bằng env NLI_HINT_ENABLED=0)

SUPPORTED_SECTOR_STANDARDS = {"GRI 11: Oil and Gas Sector 2021"}
```

### Bảng ánh xạ variant ↔ flag NLI (TN1)


| Variant | NLI_REORDER | NLI_HINT | Mô tả                         |
| ------- | ----------- | -------- | ----------------------------- |
| `a0`    | 0           | 0        | Baseline — không augmentation |
| `a1`    | 1           | 0        | Chỉ REORDER                   |
| `a2`    | 0           | 1        | Chỉ HINT                      |
| `v_new` | 1           | 1        | Cả REORDER lẫn HINT           |


