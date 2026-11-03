# MBG AI Model: Master Plan

**Owner:** Ankit (sole builder) · **Started:** 31 Oct 2026 · **Status:** Level 1, data preparation (steps 1–3 done on export `messages_2026-10-31`)

> Manager decision (31 Oct 2026): skip the internal company-support bot. Build directly for **Chatrace customers**: one MBG model plus a separate memory (RAG) per customer, served as an API that the Chatrace developer plugs in instead of GPT and other paid APIs.

---

## 1. The idea in simple words

We build **one AI model of our own**, plus **a separate memory for each Chatrace customer**. Chatrace calls **our API** instead of GPT. When someone messages a business on WhatsApp, our model answers **only from that business's memory**.

### The three things we build

| # | What | Simple meaning |
|---|---|---|
| 1 | **MBG model** | A free open model (Qwen 7–8B, Apache-2.0) that we **fine-tune**. We don't build a model from zero. The fine-tune teaches **how to behave**: short, polite WhatsApp replies in English, Hindi or Hinglish; answer only from the memory; never invent prices or promises; say "let me connect you to the team" when unsure. One model serves all ~3,000 customers. |
| 2 | **Memory (RAG)** | Each customer pastes their business info (products, prices, timings, address, FAQs) into a **Memory** section in Chatrace. We split it into small pieces and store them in **that customer's own box** (Postgres + pgvector). Customer A never sees B's data. Edits apply instantly, with no retraining. The form also has a **hand-over message** field: each business can set its own (e.g. "Call us at …", "Visit our store between 10am and 8pm"). The model uses theirs when set, otherwise the default message in the customer's language. |
| 3 | **MBG AI API** | **Chat API** ("customer X, this message, the last few messages → reply") and **Memory API** ("save / update / delete customer X's memory"). Given to the Chatrace developer. OpenAI-compatible, so very little work on their side. |

### What happens when a WhatsApp message arrives

1. A user messages **Sharma Jewellers** on WhatsApp: *"gold ring ka rate kya hai?"*
2. Chatrace calls our **Chat API** with: customer ID, the message and the last 5–10 messages.
3. We search **only Sharma Jewellers' memory** and pick the best 3–4 pieces.
4. Those pieces and the question go to **our model**, which writes the reply.
5. We return the **reply** plus a **`needs_human`** flag.
6. Chatrace sends the reply on WhatsApp, or moves the chat to the business's live agent when `needs_human` is true.

### What we need from the Chatrace developer

**Must have**
1. An **"MBG AI" option** in the model picker that calls our API instead of OpenAI.
2. **Customer ID (workspace/bot ID) on every request.** This is the most important item, because it tells us which memory to use.
3. **Conversation ID plus the last 5–10 messages** on every request.
4. A **Memory section** in the customer panel (text box, file upload later) that calls our Memory API on save. The current "write about your company" box can be reused.
5. **Handle `needs_human`**: route the chat to that business's live-agent inbox, or send their fallback message.
6. **Server-to-server calls only.** The API key never reaches a browser.

**Data and information** (see section 3 for why each matters)
- The **AI bot settings table**: each business's knowledge/prompt text (the current "about company" field) per `uid`.
- The **`chats` table** (group vs one-to-one, contact vs internal) and the **`users` table** (business name and category).
- Which **AI provider and model** each business uses today, and the prompt Chatrace sends.
- **Volume**: AI replies per day and month, and peak per minute. This decides GPU vs VMs.
- Written **permission/consent** to use anonymised conversations for training.

**Nice to have later:** per-customer bot settings (name, tone, language, hours), and usage display and limits per plan.

---

## 2. Infrastructure decisions

| Topic | Decision |
|---|---|
| Base model | **Qwen 7–8B Instruct** (Apache-2.0) first; **Llama 3.1 8B** as the alternative. Final pick after the Level 2 bake-off. |
| Fine-tuning method | **QLoRA** with **Unsloth + TRL** |
| Training hardware | **Google Colab Pro (bought 2 Nov 2026, 100 compute units): L4 by default**, A100 only for the full fine-tune if needed. Before that: FREE T4 (16 GB). Short sessions and daily GPU limits, so every job (judge, memory building, fine-tune) is split into shards/steps that are checkpointed to the **Hugging Face Hub** (private, user `ankit-iiitdmj`) and resumed after a disconnect. Models are kept 7B-class in 4-bit (AWQ for inference, QLoRA for training). |
| Storage of data and models | **Hugging Face Hub, private** repos only: dataset repo + model repo (+ GGUF repo) |
| Embeddings / reranker | **BGE-M3** (multilingual) / **bge-reranker-v2-m3**. Both run on CPU. |
| Memory store | **Postgres + pgvector**, one namespace per customer (`tenant_id`) |
| Serving (build + pilot) | Our VMs: **llama.cpp / Ollama**, 4-bit GGUF, OpenAI-compatible endpoint |
| Serving (full rollout) | **vLLM on a GPU** (likely needed for ~3,000 customers). Same API, so Chatrace sees no change. |
| API | **FastAPI**: `/v1/chat/completions`-style Chat API + Memory API |

---

## 3. What the Chatrace export contains (checked 31 Oct 2026)

File: `messages_2026-10-31.dump`. This is a PostgreSQL 18 custom-format dump of database `chatbot_db_main`, about 400 MB, containing **one table: `public.messages`**.

Columns: `id, uid (business), chat_id (conversation), type, message_id, status, body (JSON), route (INCOMING/OUTGOING), timestamp, createdAt, updatedAt, error`.

| Measure | Value |
|---|---|
| Messages | **2,992,736** (13 Jun → 31 Oct 2026) |
| Businesses (`uid`) | **1,289** |
| Conversations (`chat_id`) | **404,208**; only **144,314** have messages both ways |
| Text messages | **1.78 M** (median 35 characters); the rest are images, media, templates, errors |
| Direction | 60% outgoing / 40% incoming |
| Question → **AI** reply pairs | **~60,300** from **1,014** businesses (`body.chatbot` set, no `flow_node_id`) |
| Question → flow reply pairs | ~15,200 |
| Question → unlabelled (mostly human) pairs | ~236,700. Very noisy: staff, internal and personal chats. |

### Verdict: useful raw material, not enough by itself

**Good for:**
- **The real distribution of customer questions** in Hinglish, Hindi and English, across many business types. This is the most valuable part.
- **The evaluation set**: real questions to test every model version on.
- **Candidate training replies**: the ~60k AI replies, after filtering, *if the provider's terms allow it* (see the risks below).

**Missing or problems:**
1. **No business memory.** The knowledge or prompt each business had when the AI replied isn't in this table. Without it we can't build *(memory, question → answer)* training examples. **We must get the AI bot settings table from Chatrace.**
2. **No conversation context type.** We can't tell group from one-to-one, or customer from internal staff chat. **We need the `chats` table.**
3. **Provider terms.** The AI replies were generated by GPT or another paid model. Some providers forbid using their outputs to train a competing model. **Check the terms of the provider Chatrace used before training on those replies.** The safe alternative is to use the real questions + memories and generate the replies with a large *open* model (Apache-licensed teacher), then human spot-check them.
4. **Personal data everywhere** (phones, emails, names, addresses). It must be redacted **before** upload to Colab or Hugging Face. Hugging Face repos stay **private**. Get written consent/permission (India DPDP Act 2023).
5. **Noise**: error messages ("AI reply is temporarily unavailable…"), single-word replies, emoji-only messages, lead lists, spam. All of it needs filtering.

> Keep the `.dump` file out of git and off shared drives. Delete local extracts once the cleaned, redacted dataset is on the private Hugging Face repo.

---

## 4. Step-by-step plan, level by level

Each level ends with **Done when**. Don't start the next level until it passes.

### Level 0: Groundwork and approvals (≈ 1 week, partly in parallel)
- [ ] Manager/legal OK to use anonymised Chatrace conversations for training (consent, DPDP).
- [ ] Check the terms of the AI provider(s) Chatrace used: may their outputs be used for training?
- [ ] Ask the Chatrace developer for: **AI bot settings table** (knowledge/prompt per `uid`), **`chats` table**, **`users` table** (name, category), current prompt format, provider and model per business, volume numbers.
- [ ] Create private Hugging Face repos: `mbg-ai/chat-dataset` (dataset), `mbg-ai/qwen-mbg-lora` (checkpoints/adapter), `mbg-ai/qwen-mbg-gguf` (quantised).
- [ ] Colab notebook skeleton: mount Drive, `huggingface_hub` login with a token stored in Colab **Secrets** (never hard-coded).
- **Done when:** permission is written down, the provider-terms answer is known, and the extra tables are requested.

### Level 1: Data preparation (≈ 1–2 weeks)

**Progress, export `messages_2026-10-31`** (scripts in `src/data/`, reports in `logs/`):

| Step | Script | Result |
|---|---|---|
| ✅ 1 Extract | `01_extract_conversations.py` | 2,992,736 rows → 1,867,498 text messages → 339,126 conversations (96 s) |
| ✅ 2 Clean | `02_clean_filter.py` | 103,119 conversations · 1,232 businesses · 332,229 answered questions (AI 79,808 · staff 227,460 · flow 35,349 · bot/agent 8,965) |
| ✅ 3 Redact | `03_redact_pii.py` | Customer: 25,264 phones, 7,933 emails, 19,123 IDs, 44,250 names removed. Business: 97,457 phones + 5,724 emails → consistent fake values. Verified: 0 mobiles / 0 emails left in 361,198 customer turns. IDs pseudonymised. |
| ⏳ 3b Spot-check | `redaction_samples.txt` | Ankit to read ~200 samples by hand |
| ✅ 3c Quality audit | `quality_audit.py` | 332,229 pairs → rule-clean 157,614 (47%). Reading samples shows most rule-clean pairs are still **staff/personal chats**. Questions that look like real customer enquiries: ~18k AI-answered (805 businesses) + ~28k staff-answered (768). Top canned replies: "ok" (6,177), distributor/brochure auto-replies (1–3k each). **Merge bug found:** 6,314 AI replies glued to later staff messages → must fix. |
| ✅ 4 Gates 1+2 → candidates | `04_build_candidates.py` | Gate 2 merge fix (reply cut at first message from another author) + Gate 1 rules (greeting, low-info, canned, dead-end, too short/long, **bot-to-bot** 3,264) + flow menus dropped + staff pairs only if the question looks like an enquiry → **82,175 candidates** (AI 50,743 · staff 27,012 · bot 3,701 · agent 719) from 1,099 businesses |
| ✅ 5 Push candidates | `05_push_candidates.py` | → private HF dataset `ankit-iiitdmj/mbg-chat-candidates` (`--file` pushes other redacted files, e.g. evidence) |
| ✅ Gate 3 AI judge | `notebooks/01_judge_candidates.ipynb` + `src/judge/` | Colab free T4, Qwen2.5-7B-Instruct-AWQ (vLLM), ~6.5 h, shards in `ankit-iiitdmj/mbg-judge-<export>`. **82,175 judged · 5 parse failures · keep_question 63,436 · keep_pair 33,188 (949 businesses)**. Questions: Hinglish 32.7k · English 29.6k · Hindi (Devanagari) 0.9k · Marathi 26. Top topics: booking, price, product info |
| ✅ Gate 4 review | sample of 200 (120 keep_pair · 40 question-only · 40 dropped), reviewed by Claude for Ankit | keep_pair **86% correct as-is → ~93% after the mechanical fixes below** (pass ≥ 90%). Dropped: 37/40 correct. Question-only: 27/40 correct (6 good replies judged too strictly, 7 not real customers). Issues → fixed in step 7: (a) AI/bot replies made of 2+ separate messages (3,294 keep_pairs) → drop; (b) leftover `<ID>/<PHONE>` tags in replies (891) → drop; (c) `**bold**` (20,692) → WhatsApp `*bold*`; (d) one business dominates (top 10 = 28%) → cap 200 pairs per business; (e) replies that greet the customer by name ("Prachi", "Roshan ji") → strip/skip, the model never sees the name; (f) role-reversed chats (vendor pitching the business, B2B partners) ~5% → accepted noise; (g) a password in a customer message → drop questions with password/OTP |
| ✅ 6a Collect evidence | `06_collect_evidence.py` | Fact-bearing business messages (prices, timings, address, links, offers, menus) + the question they answered, deduped, newest wins → **1,002 businesses**, 22,396 snippets, median 7k chars; pushed to HF |
| ✅ 6b Build memories | `notebooks/02_build_memories.ipynb` + `src/memory/` | Colab Pro L4, Qwen2.5-14B-Instruct-AWQ → 925 parsed, 899 good. **Fact check** (`memory/clean.py`): every link, 4+ digit number and clock time must appear in the business's own messages → removed 4,788 values (≈550 invented Google-Maps links, sample phone numbers, "Not specified", 138 invented timings) → **897 good memories** |
| ✅ 7a Build items | `07_build_items.py` | **19,924 items from 892 businesses** (≤ 40 per business, ≤ 150 copies of one question, password/OTP questions dropped). Prompt = production system prompt with header + top-3 chunks (+ 1 unrelated chunk 35% of the time) + hand-over (business's own 70%, else default in the customer's language) + last turns. 18,318 teacher items + 1,606 synthetic hand-overs (answering chunks removed). Languages: Hinglish 11.4k · English 7.6k · Hindi 0.9k |
| ⏳ 7b Teacher replies | `notebooks/03_generate_answers.ipynb` | Colab Pro L4, Qwen2.5-14B-Instruct-AWQ writes the ideal reply from the same prompt + `TEACHER_GUIDE` → `ankit-iiitdmj/mbg-answers-<export>` |
| ⬜ 8 Check + split + push | `08_assemble_split_push.py` | Reject replies with facts not in their chunks (links, 3+ digit numbers, times), tags, customer names, wrong script, > 700 chars; hand-overs normalised to the exact message; split by business 90/5/5 → `ankit-iiitdmj/mbg-sft-<export>` |

1. **Extract** text conversations from the dump (stream with `pg_restore -a -t messages`; no need to restore a DB).
2. **Rebuild conversations** per `chat_id` in time order, and merge consecutive messages from the same side.
3. **Label who replied**: AI (`body.chatbot`), flow (`flow_node_id`), human (`sentBy`/unlabelled), template.
4. **Filter**:
   - drop error/limit messages, empty, emoji-only, under 3 characters, media placeholders
   - drop group/internal chats (once the `chats` table arrives)
   - drop spam
   - keep chats with a real customer question
5. **Redact PII**: phone numbers, emails, names, order IDs, URLs with personal parameters → placeholders (`<PHONE>`, `<EMAIL>` …). Spot-check 200 rows by hand.
6. **Join business memory** (from the AI bot settings table) to each conversation by `uid`. Chunk it the same way the live RAG will.
7. **Build training examples** in chat format:
   - `system`: MBG assistant rules
   - `context`: top-4 memory chunks *as retrieval would return them* (include 1–2 irrelevant chunks sometimes, for robustness)
   - `history`: last turns
   - `user`: the question
   - `assistant`: the ideal reply
   - Ideal replies come from: filtered AI replies (only if the terms allow) **or** replies generated by an open teacher model from the same memory, then quality-filtered.
   - Add **"not in memory → hand over"** examples (≈10–15%) and **language-match** examples (Hinglish in → Hinglish out).
8. **Balance**: cap the examples per business (so the top 10 businesses, 16% of messages, don't dominate), and mix business types.
9. **Split by business** (not by row): train 90% / validation 5% / **golden test set** 5%, plus 300 hand-checked golden questions with expected answers.
10. **Push** to the private HF dataset repo (JSONL, versioned: `v1`, `v2`…).
- **Done when:** ~10–20k clean examples are on Hugging Face, 200 random rows show zero personal data, and the golden set is reviewed.

### Level 2: Baseline with no fine-tune (≈ 1 week)
- [ ] Minimal RAG prototype (pgvector + BGE-M3 + reranker) on 10 sample businesses' memories.
- [ ] Measure **retrieval recall@4** on the golden set. Target **≥ 90%**; fix chunking before going on.
- [ ] Run **base Qwen 7–8B and Llama 3.1 8B** (4-bit, Ollama on a VM) with RAG on the golden set.
- [ ] Score: correct from memory, made-up facts, correct hand-over, language match, length, speed on the VM.
- **Done when:** the model choice is made and the baseline scores are recorded (the fine-tune must beat them).

### Level 3: Fine-tune on Colab T4 (≈ 1–2 weeks across sessions)
**Setup (T4 = 16 GB, no bf16):**
- Unsloth `FastLanguageModel.from_pretrained(<qwen-7/8b-instruct>, load_in_4bit=True, max_seq_length=2048)`
- LoRA: `r=16`, `alpha=16`, dropout 0, target `q,k,v,o,gate,up,down`, `use_gradient_checkpointing="unsloth"`
- Train: `per_device_train_batch_size=2`, `gradient_accumulation_steps=8`, `learning_rate=2e-4`, `num_train_epochs=2–3`, `fp16=True`, `optim="adamw_8bit"`, `warmup_ratio=0.03`, `lr_scheduler="cosine"`
- Train only on the **assistant** turns (mask system, context and user).

**Never lose progress:**
- `save_strategy="steps"`, `save_steps=100–200`, `save_total_limit=3`, **plus** a save at every epoch end.
- `push_to_hub=True`, `hub_model_id="mbg-ai/qwen-mbg-lora"`, `hub_strategy="checkpoint"`, `hub_private_repo=True`. The latest checkpoint is always on Hugging Face.
- Also copy `output_dir` to Google Drive.
- **Resume** in a new session: download the latest checkpoint from the Hub (or Drive), then `trainer.train(resume_from_checkpoint=True)`.
- Log to a run sheet: session #, start step, end step, loss, eval loss.
- Rough time on T4: about 1.5–2k tokens/s, so 10k examples × ~600 tokens ≈ 1 hour per epoch. Plan for several sessions. *(Estimate only; measure in session 1.)*

**After each epoch:** evaluate on the validation set and 50 golden questions. Stop early if eval loss rises or golden scores drop.
- **Done when:** the best checkpoint beats the Level 2 baseline on the golden set with no increase in made-up facts.

### Level 4: Evaluation and sign-off (≈ 3–5 days)
- [ ] Full golden-set run: fine-tuned + RAG vs base + RAG (vs current GPT replies for reference).
- [ ] Human review of 200 answers by the support/onboarding team: correct / acceptable / wrong / hallucinated.
- [ ] Red-team: prompt injection ("ignore your rules"), rude users, off-topic, competitor questions, price fishing.
- **Done when:** the agreed quality bar is met (e.g. ≥ 85% correct/acceptable, ≤ 2% made-up facts, ≥ 90% correct hand-overs).

### Level 5: Package and deploy for testing (≈ 3 days)
- [ ] Merge LoRA → export **GGUF Q4_K_M** (Unsloth `save_pretrained_gguf`) → push to `mbg-ai/qwen-mbg-gguf`.
- [ ] Run on a VM with **llama.cpp server / Ollama** (OpenAI-compatible endpoint).
- [ ] Load test: replies per minute, time to first word, full reply time, RAM.
- **Done when:** the model serves from a VM and its speed is measured.

### Level 6: Memory service + APIs (≈ 2 weeks)
- [ ] **Memory API**: `POST/PUT/DELETE /v1/memory/{customer_id}`. Chunk → embed (BGE-M3) → pgvector with `tenant_id`; re-embed on update.
- [ ] **Chat API**: OpenAI-style `/v1/chat/completions` with `customer_id`, `conversation_id` and messages. Rewrite query → hybrid search (vector + keyword) → rerank → score threshold → model → reply + `needs_human` + `sources`.
- [ ] **Isolation**: every query is filtered by `tenant_id`; test that customer A can never get B's chunks.
- [ ] **Guardrails**: answer only from memory, refuse off-topic, hand-over topics (refund, complaint, payment failure), prompt-injection filter.
- [ ] **Hand-over message**: when `needs_human` is true, reply with the business's own hand-over message from its memory form if set; otherwise the default in the customer's language (EN: "Thank you for your query! Our team member will connect with you shortly with more details." · Hinglish: "Aapke sawal ke liye dhanyavaad! Hamari team ka member jaldi hi aapse connect karega." · Hindi: "आपके सवाल के लिए धन्यवाद! हमारी टीम का सदस्य जल्द ही आपसे संपर्क करेगा।").
- [ ] **Per-customer API usage metering**, rate limits, logs (redacted).
- [ ] Fallback to a third-party API if our model server is down (optional, by decision).
- **Done when:** the APIs pass tests with 10 sample businesses, including isolation tests.

### Level 7: Chatrace integration (≈ 1–2 weeks, with the Chatrace developer)
- [ ] Hand over the API spec + keys (staging).
- [ ] The Chatrace developer adds the "MBG AI" option, sends customer ID + history, builds the Memory section, handles `needs_human`.
- [ ] End-to-end test on a test WhatsApp number in staging.
- **Done when:** a real WhatsApp message gets a correct MBG AI reply through Chatrace in staging.

### Level 8: Pilot (≈ 2–4 weeks)
- [ ] 10–20 real customers of different business types and languages.
- [ ] Daily review of a sample of replies; customers rate replies (thumbs up/down).
- [ ] Watch speed and errors; decide on a GPU with these rules: **> 5 s to first word at peak**, **> 3 chats queued regularly**, or **> 30 s per full reply** for a week.
- **Done when:** pilot quality holds for 2 weeks and the GPU decision is made.

### Level 9: Rollout and improvement loop (ongoing)
- [ ] GPU + **vLLM** if needed (same API).
- [ ] Gradual rollout: 10% → 50% → all customers who pick MBG AI.
- [ ] Monthly: collect rated replies → fix → retrain (back to Level 3) → re-evaluate → swap model version with rollback.

---

## 5. Open questions / waiting on

| # | Question | Who | Status |
|---|---|---|---|
| 1 | Permission to use anonymised conversations for training | Manager | ✅ Granted (31 Oct 2026); more exports coming |
| 2 | Do the current AI provider's terms allow training on its outputs? | Ankit / Manager | Open |
| 3 | AI bot settings table (knowledge/prompt per `uid`) | Chatrace developer | ➖ Not required. We design our own memory format (decision 31 Oct); old table is optional extra |
| 4 | `chats` + `users` tables | Chatrace developer | Open |
| 5 | Daily/monthly AI reply volume, peak per minute | Chatrace developer | Open |
| 6 | Current prompt format + provider/model per business | Chatrace developer | Open |
| 7 | VM specs (vCPU, CPU model, RAM) | Ankit | Open |

---

## 6. Decision log

| Date | Decision |
|---|---|
| 2026-10-23 | Use RAG for facts + fine-tune for behaviour (never fine-tune facts). |
| 2026-10-23 | Model family: Qwen 7–8B first, Llama 3.1 8B alternative. Final pick after the Level 2 bake-off. |
| 2026-10-31 | **Manager: skip the company-support bot; build directly for Chatrace customers.** |
| 2026-10-31 | Training on Colab T4 with checkpointing to Hugging Face Hub (private). |
| 2026-10-31 | Chatrace export checked. Useful for questions and evaluation; business memory table + consent + provider-terms check still needed. |
| 2026-10-31 | Manager granted full permission to use the exports; more exports coming. |
| 2026-10-31 | Hand-over wording (default, language-matched): EN "Thank you for your query! Our team member will connect with you shortly with more details." · Hinglish "Aapke sawal ke liye dhanyavaad! Hamari team ka member jaldi hi aapse connect karega." · Hindi "आपके सवाल के लिए धन्यवाद! हमारी टीम का सदस्य जल्द ही आपसे संपर्क करेगा।" Each business can override it via a "hand-over message" field in the memory form. |
| 2026-10-31 | Training size: pick the best ~10–20k examples (quality over quantity); extra exports add business-type variety and test questions, not bulk. |
| 2026-10-31 | **No training on raw replies.** Real data is used mainly for the *questions*. Training answers are written from the business memory (teacher model); an original reply is kept only if the AI judge rates it good and consistent with the memory. Greeting examples ("hi" → proper welcome) are generated well and capped at ~3%. Canned auto-replies, "number closed / call on…" replies, staff/personal chats are excluded. |
| 2026-10-31 | **We design our own structured memory format** (name, category, services + prices, timings, location, contact, FAQs, policies, never-say, tone/language). The dashboard team builds the Memory form to match. Training memories are written in this format per business (partly extracted from each business's chats) and paired with its real customer questions. The old Chatrace AI-prompt table is not required. |
| 2026-11-02 | Gate 3 finished on the free T4 (82,175 judged). Manager bought **Colab Pro**; use the **L4** (~5 units/h) for steps 6–8 and training; A100 only for the full fine-tune. |
| 2026-11-02 | Gate 4 done by Claude on Ankit's request (no time for manual review): pass after fixes. Step 7 applies the fixes listed in the Gate 4 row. |
| 2026-11-02 | Memory teacher = **Qwen2.5 Instruct (Apache-2.0)**, size chosen by GPU (L4 → 14B). Memories are extracted only from each business's own messages; a business's usual "call us for details" text becomes its `handover_message`. |
| 2026-11-03 | **Training replies are written by the open teacher (Qwen2.5-14B, Apache-2.0), never copied from the chats.** AI/bot replies are third-party model outputs (provider terms); staff replies often state facts that are not in the memory (dates, promises), which teaches making things up — so `USE_HUMAN_REPLIES = False`. Real data supplies the questions, history and the facts in the memory. |
| 2026-11-03 | Memories and replies are fact-checked mechanically: every link, longer number and clock time must appear in the source (business messages for memories, the given chunks for replies). |
