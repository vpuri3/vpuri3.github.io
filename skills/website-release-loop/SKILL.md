---
name: website-release-loop
description: Build, preview, and publish the Hugo site in vpuri3.github.io after content or style edits. Use when working on files under content/, assets/, layouts/, static/, or hugo.toml and you need to (1) run a clean build, (2) render locally at localhost:1313, and (3) deploy to GitHub Pages via push.
---

# Website Release Loop

Use this workflow after every website change.

## Quick Commands

- Validate build: `bash skills/website-release-loop/scripts/release_loop.sh check`
- Preview locally (foreground): `bash skills/website-release-loop/scripts/release_loop.sh preview`
- Keep server alive in background: `bash skills/website-release-loop/scripts/release_loop.sh start`
- Check background status: `bash skills/website-release-loop/scripts/release_loop.sh status`
- Read recent logs: `bash skills/website-release-loop/scripts/release_loop.sh logs`
- Restart background server: `bash skills/website-release-loop/scripts/release_loop.sh restart`
- Stop background server: `bash skills/website-release-loop/scripts/release_loop.sh stop`
- Publish (commit + push): `bash skills/website-release-loop/scripts/release_loop.sh publish "your commit message"`

## Workflow

1. Run clean build.
2. Fix any build or rendering issues.
3. For manual editing sessions, use `start` so the server survives agent session changes.
4. Review at `http://localhost:1313/`, monitor with `status`/`logs`, and use `restart` if needed.
5. Push to `master` to trigger `.github/workflows/Deploy.yml`.

## Guardrails

- Keep commit scope intentional; do not blindly include unrelated files.
- Do not publish until user approves push.
- Prefer background `start` for long editing sessions; avoid relying on one-off foreground sessions.

## Notes

- Hosting is GitHub Pages via deployment workflow on push.
- If deployment status is needed, run `gh run list --workflow Deploy.yml --limit 1`.
