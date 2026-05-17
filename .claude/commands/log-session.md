---
description: Aggiorna il vault Obsidian con un log della sessione corrente.
allowed-tools: [Read, Write, Edit, Glob, Grep, Bash]
---

End-of-session wiki update.

Use the `thesis-obsidian` skill (or follow the LLM Wiki pattern documented
in `Vault Obsidian/Test Second Brain/CLAUDE.md` directly if the skill is
not available in this session).

Goal:

1. Append a new dated entry to
   `C:\Users\Feder\OneDrive\Desktop\TESI\Vault Obsidian\Test Second Brain\wiki\log.md`.
   Format:

   ## YYYY-MM-DD — short title

   - bullet 1, max 25 words
   - bullet 2
   - bullet 3 (3 to 6 bullets total, not a transcript)

   Commits: short hashes from `git log` of the Fasi Applicative repo
   since the last log entry.
   Files modified outside the codebase: list.
   Open questions for Claude Chat: list (may be empty).

2. If any commit was made under `Fasi Applicative/` since the last log
   entry, append a one-line entry to the matching
   `wiki/thesis/phase-logs/phase<N>-log.md` for each Phase touched.
   Parse the commit scope prefixes: phase1, phase2, phase3, phase4. If
   `phase-logs/phase<N>-log.md` does not exist, create it with the H1
   header `# Phase <N> log` and the new line.

3. Do NOT edit anything under `Fasi Applicative/`. This command writes
   only to the Obsidian vault.

Rules: no em-dash, no Oxford comma. Use English.

When done, print exactly:
`WIKI LOG UPDATED — <YYYY-MM-DD> — <N> entries appended`
