# Personal AI Framework — Feature Roadmap

Running list of planned features. Status as of **v5.9.2** (2026-04-18).

---

## ✅ Completed

### Quality of life / polish (v5.7.0)
- **Streaming TTS** — sentence-by-sentence playback while the model is still generating, so speech starts immediately instead of waiting for the full response.
- **Conversation export** — export any single conversation to Markdown from the sidebar.
- **Light mode + theme toggle** — full light theme with persisted preference.

### Smarter retrieval / context (v5.7.0)
- **Multi-document synthesis** — answers blend evidence from multiple retrieved chunks rather than quoting one source.
- **Citation links with score breakdown** — each citation shows semantic / BM25 / rerank scores so you can see *why* a chunk was chosen.
- **Conversation-aware RAG** — retrieval uses the last 6 turns as context and boosts chunks that match the ongoing topic.

### Agent capabilities (v5.8.0)
- **Google Calendar integration** — `calendar_service.py` + six endpoints (`/calendar/status`, `/auth-url`, `/oauth-callback`, `/disconnect`, `/events`, `/create`). Reuses `gmail_credentials.json`, separate `calendar_token.pickle`. Settings UI has Connect/Disconnect. Natural-language "add this to my calendar" lands with tool-calling below.
- **Write-capable tools** — `write_file` and `run_script` added to `agent_tools.py`, sandboxed to `~/.personal-ai/skippy-workspace/`. Confirm-tool registry + two-phase flow: LLM proposes → UI shows approve/deny card → `/agent/execute` runs after approval. Path traversal blocked, 100 KB write cap, 30s script timeout, only `.py` and `.sh` scripts allowed.
- **Reminders** — `reminder_service.py` (SQLite at `~/.personal-ai/reminders.db`) + `schedule_reminder` confirm-tool. Endpoints: `/reminders` (create/upcoming/due/ack/cancel). UI polls `/reminders/due` every 30s, surfaces via toast + system bubble + browser Notification API. Detection prompt gets current datetime so Qwen can convert "tomorrow at 9am" into ISO 8601.
- **Tool registry: Email + Calendar** — `send_email` and `create_calendar_event` as confirm-tools; `list_calendar_events` as safe/auto-tool. Detection prompt has few-shot examples for each. Skippy can now draft+send email, add calendar events, and answer "what's on my calendar" from chat.
- **Collapsible tool trace UI** — each assistant message renders a small `🔧 {tool name} ▸` card above the response; expand reveals params + raw tool result. `_build_tool_context` rewritten to wrap results in `<toolname_output>…</toolname_output>` tags with strict "don't echo the scaffolding" instructions. Safety-net post-processor (`cleanAssistantText`) strips any `TOOLS`/`[planning]`/`TOOL RESULT` blocks the model still leaks. Fixed streaming-text bug where `innerHTML =` on every token was wiping inserted trace cards.
- **Multi-step agent loop** — `_run_agent_loop()` runs up to 5 iterations of safe-tool detection + execution, feeding each result back into the next detection pass via `tool_history`. UI gets one `tool_use` SSE event per step → multiple stacked trace cards. A confirm-tool mid-chain breaks the loop and surfaces as a pending-action card with a "After running N prior step(s)" prefix. `_build_tool_chain_context()` injects all step results into the final-generation system context.
- **Agent polish round** (same-day follow-up):
    - Robust tool-call parser — slice-based extraction of first `<params>…</params>` block, survives multi-tool emissions, survives JSON content with arbitrary braces.
    - Detection pass `max_tokens` 512 → 2048, removed `\n\n\n` stop that truncated multi-paragraph bodies. Long emails/files now compose cleanly.
    - `send_email` no longer produces `"Subject: …"` subjects or raw-paste bodies — tightened description + extra chained-composition few-shot example.
    - `write_file` content no longer gets a stray `Subject: …` first line — prompt-level fix + server-side scrubber in `_check_for_tool_use` (so UI preview matches what will be written) + defense-in-depth scrubber in `tool_write_file`.
    - Pending-action card redesigned: each param rendered as its own row; long string values (>400 chars or >6 lines) show a 6-line preview with a `▸ Show full {field} (1.2 KB)` collapsible expander instead of scroll-dominating the chat.
    - Overwrite safety — server attaches `meta.existing_file {size, mtime}` to `write_file` pending actions. UI shows an orange "⚠️ This will overwrite an existing file" banner and replaces the two-button row with three options: **⚠️ Overwrite** / **📄 Save as copy** / **❌ Deny**. "Save as copy" hits a new `/agent/next-free-path` endpoint that returns `{stem}_01.ext`, `_02`, …
    - On approve, the warning banner morphs to reflect final state: overwrite → green ✅ "Overwrote previous version (was 1.8 KB)"; save-as-copy → banner removed.
    - Removed stale welcome-screen suggestion pills (`i.MX demo email`, `ARM projects`, `Delay follow-up`).

### Training / model quality (v5.8.0)
- **RLHF-lite** — `feedback_service.py` + SQLite `feedback` table (shares `conversations.db`). Endpoints: `POST /feedback`, `DELETE /feedback/{message_id}`, `GET /feedback/{message_id}`, `GET /feedback/stats`, `GET /feedback/history?days=30`. UI renders 👍/👎 buttons below every assistant message; click to rate, click same to clear, click opposite to flip. Rating overwrites on conflict (`ON CONFLICT(message_id) DO UPDATE`). `conversation_store.add_message` now returns the new `message_id` so the frontend can wire buttons to the correct row.
- **Training dashboard** — new 📊 header button opens a modal with: 👍/👎 counts + up rate, last-30-days bar chart (daily buckets, green up / red down), list of LoRA checkpoints read from `training/output/*/trainer_state.json` with inline SVG sparkline of train loss per run. Endpoints: `GET /training/runs`, `GET /training/run/{name}`. docker-compose mounts `./training:/app/training:ro` so the container can read checkpoint logs.
- **Selective training data** — `conversations.excluded_from_training` column with auto-migration for older DBs. Sidebar gets a 🎓 / 🚫 toggle per conversation (strike-through + opacity when excluded). Endpoints: `POST /conversations/{id}/exclude-from-training`, `DELETE` to unset. `training/collect_training_data.py` skips excluded conversations and any user→assistant pair where the assistant message has 👎 rating — closes the RLHF → training feedback loop.
- **Tool-call bleed scrubber** — `cleanAssistantText` also strips standalone tool-invocation lines like `write_file(path="foo.md", content=content)` or `send_email(to="x@y", …)` that Qwen sometimes echoes into final prose.

### Observability (v5.8.0)
- **Prometheus / Grafana metrics** — `pipeline/metrics.py` defines 13 series (generations/latency/TTFT/throughput, tool calls by tool+outcome, feedback by rating, reminders created/fired, RAG queries by mode + docs-returned, model_loaded/maintenance_mode/knowledge_base_docs gauges). `/metrics` endpoint on the FastAPI server renders text format 0.0.4. Instrumented both `/generate` and `/generate/stream` (endpoint label distinguishes them). `monitoring/` has `prometheus.yml` (scrapes `host.docker.internal:8080`), `grafana-dashboard.json` (13 panels with p50/p95/p99 percentiles), and a README with one-liner docker commands for Prometheus (port 9090) + Grafana (port 3001).

### Multi-user + mobile (v5.9.0 – v5.9.2)
- **Multi-user support** (v5.9.0 / v5.9.1 polish) — per-user data isolation under `~/.personal-ai/users/<username>/` (conversations.db, reminders.db, settings.json, memory/facts ChromaDB collections name-suffixed, skippy-workspace, Gmail/Calendar tokens). bcrypt password auth via `user_service.py` + sessions table; bearer-token in localStorage (chosen over cookies because of cross-origin HTTP quirks on LAN). FastAPI middleware gates all non-public paths; `require_admin` for user-management endpoints. OAuth callbacks stay public by encoding username into the OAuth `state` parameter. First-run bootstrap flow creates the admin account; legacy single-user install migrates into `users/kyle/` automatically on first startup. Web UI has login overlay, Account section in Settings with change-password + sign-out, admin-only Users management prompt.
- **Auth middleware / CORS preflight fix** (v5.9.2) — middleware short-circuits OPTIONS preflight so browser cross-origin requests don't trip auth before CORS headers are applied.
- **Mobile-friendly UI** — responsive `@media (max-width: 768px)` breakpoint: sidebar becomes a slide-in drawer (☰ hamburger + backdrop + auto-close on conversation select), header buttons collapse to icons, conversation actions always visible (no hover on touch), messages span full width, textarea uses 16px font so iOS doesn't auto-zoom. API base now derived from `window.location.hostname` so the frontend works from any device on the LAN without code changes.
- **Phone Access helper** — new Settings → 📱 Phone Access panel. `./run.sh start` writes the host's LAN IPs to `~/.personal-ai/lan_ips.txt`; `/system/lan-ips` reads + filters out docker/loopback addresses. One-click **🔎 Detect LAN IP** auto-fills the URL. `/system/qr?data=...` returns a server-rendered SVG QR code (via the `qrcode` pip package). Includes firewall-troubleshooting hints for `ufw`.

### Base model + cloud training (on `main`, post-v5.9.2)
- **Kyle-merged Qwen3-30B-A3B MoE deploy** — swapped base from Qwen 2.5 14B Instruct to Kyle-merged Qwen3-30B-A3B MoE (Q4_K_M). Production decode on 5090: 155 tok/s sustained / 192 peak plain-prose, 69.7 tok/s tool-use path, 14.7 tok/s cold start.
- **Cloud training pipeline hardening** — MoE LoRA training works end-to-end on a RunPod H100; first real run completed successfully (~$15 / 5 h). Runbook at `docs/cloud-training-runbook.md`.
- **Use-case deck generator** — `scripts/build_use_case_deck.py` regenerates the 16-slide NPU sizing + use-case PowerPoint deck from source data.
- **Workload distribution benchmark** — `bench_skippy.py` hits `/generate/stream` across 5 workload categories (plain_chat / long_form_reasoning / tool_use / rag_long_context / cold_start) and reports TTFT + decode tok/s percentiles. Measured spread: 3.6 → 222 tok/s decode (~60× range) across real workloads — fed into the Keyhole sizer as independent validation.

---

## 🔨 Remaining

(All shipped 🎉 — add a new section here for the next round.)

---

## Notes
- Source of truth: this file. Memory index (`MEMORY.md`) points here so any new session can pick up the list.
- When a feature ships, move it from **Remaining** to **Completed** with its release version tag.
