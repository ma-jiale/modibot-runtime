# Project Agent Instructions

## Comment style

- Use CS61A-style comments and docstrings: explain what a function or class does, its important inputs/outputs, and any key invariant.
- Prefer concise docstrings on public classes/functions and short comments before non-obvious logic.
- Do not comment obvious assignments, imports, or simple one-line control flow.
- Keep comments in English ASCII unless the surrounding file already uses another language safely.
- When a function has a tricky side effect, state it directly. Example: "Append the turn only after the API call succeeds."
- Keep comments close to the code they explain, and update or remove stale comments when behavior changes.

## Secrets safety

- Never commit, stage, paste, or log API keys, tokens, private keys, `.env` contents, or credentials.
- Keep real API keys only in local environment variables or ignored local files such as `.env`.
- Do not add local test scripts that contain real keys to the repository.
- Before committing, check `git status` and staged diffs to confirm no secrets are included.
