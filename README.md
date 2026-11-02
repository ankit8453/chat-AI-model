# MBG AI Model

MBG's own WhatsApp business-assistant model for Chatrace customers: **one fine-tuned open model** (Qwen 7–8B) plus a **separate memory (RAG) per customer**, served as an API.

The full plan, decisions and open questions are in **[MASTER_PLAN.md](MASTER_PLAN.md)**. Update its checkboxes and decision log as work moves on.

## Folder layout

| Folder | What lives there |
|---|---|
| `config/` | Our memory format (`memory_schema.json`), default hand-over messages, data-prep settings |
| `prompts/` | The model's behaviour rules (system prompt) and the teacher prompts used to build training data |
| `data/raw/` | Chatrace exports (`*.dump`). **Never committed.** |
| `data/interim/` | Extracted / cleaned / redacted conversations (JSONL). **Never committed.** |
| `data/processed/` | Final training files ready for Hugging Face (JSONL). **Never committed.** |
| `src/common/` | Shared helpers (PII redaction, text cleaning, JSONL I/O) |
| `src/data/` | Data-prep pipeline, run in order `01_` → `06_` |
| `training/` | Colab T4 QLoRA training with checkpoint + resume via Hugging Face Hub |
| `eval/` | Golden test set + evaluation scripts |
| `serving/` | Model server setup (llama.cpp / Ollama now, vLLM later) |
| `api/` | Chat API + Memory API (FastAPI) |
| `logs/` | Training run log and data-prep reports |

## Data-prep pipeline (Level 1)

```bash
cd mbg-ai-model
# 1. Extract text conversations from a Chatrace dump (streams it; no database restore needed)
python src/data/01_extract_conversations.py --dump ../messages_2026-10-31.dump
# 2. Clean + filter (drops errors, noise, one-sided and internal-looking chats)
python src/data/02_clean_filter.py
# 3. Redact personal data (customer phones/emails/names → removed; business numbers → consistent fake values)
python src/data/03_redact_pii.py
# (optional) quality report of all question → reply pairs
python src/data/quality_audit.py
# 4. Gates 1 + 2: rules + clean reply cut → candidates for the AI judge
python src/data/04_build_candidates.py
# 5. Push candidates (already redacted) to the PRIVATE HF dataset ankit-iiitdmj/mbg-chat-candidates
python src/data/05_push_candidates.py
```

Then on **Colab (free T4)**: open `notebooks/01_judge_candidates.ipynb` → Runtime: T4 GPU → **Run all**.
Cell 3 asks for your HF token; results go to the private dataset `ankit-iiitdmj/mbg-judge-<export>`
shard by shard, so a disconnected session resumes where it stopped.

Steps 06–08 (build each business's memory, build training examples, split and push) come next. See MASTER_PLAN.md, Level 1.

## Rules

- Customer data stays **local** until it is redacted. Hugging Face repos are **private**.
- Keep `.env` (Hugging Face token) out of git. Copy `.env.example` → `.env`.
- Each export is processed into its own folder `data/interim/<export-name>/`, so new exports never overwrite old ones.
