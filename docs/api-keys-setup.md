# API keys for eval scripts (the durable fix)

**The recurring problem:** Claude Code's Bash tool and its `!` prompt run *non-interactive* shells
that never source `~/.bashrc`, so any `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` exported there is
invisible to every script a session runs. And the obvious workaround — putting the key in
`.claude/settings.local.json` under `env` — is a trap: it injects into the *global* Claude Code
environment and **has broken Remote Control before** (2026-05-08). Both routes are why this kept
being "sporty" across sessions.

## The fix: one gitignored file outside the repo, loaded per-process

Keys live in **`~/.personal-ai/keys.env`** — outside the git tree (cannot be committed), loaded by
`eval/_keys.py` into the *script's* process only (never the global shell env, so Remote Control is
untouched).

### One-time setup (do this once; it then works for every session)
```bash
cp ~/.personal-ai/keys.env.example ~/.personal-ai/keys.env
# edit ~/.personal-ai/keys.env and paste your real keys (from your password manager, or
# console.anthropic.com / platform.openai.com). KEY=VALUE, one per line.
chmod 600 ~/.personal-ai/keys.env
```

### Using it from a script
```python
import sys, os
sys.path.insert(0, "eval")           # or the dir containing _keys.py
from _keys import ensure_keys
missing = ensure_keys(("ANTHROPIC_API_KEY", "OPENAI_API_KEY"))
if missing:
    sys.exit(f"ERROR: {missing} not set — see docs/api-keys-setup.md")
# keys are now in os.environ for THIS process; the Anthropic/OpenAI SDKs pick them up.
```

That's it. `python3 eval/judge_agentic_tasks.py` (and any other eval script wired to `ensure_keys`)
then works from the Claude Code Bash tool directly — no interactive shell, no global env, no RC
breakage.

**Never** commit `keys.env`, paste a real key into a chat/PR, or put it in `settings.local.json`.
If a key leaks, rotate it at the provider console.
