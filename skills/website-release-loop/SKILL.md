---
name: website-release-loop
description: Build, preview, and publish the Hugo site in vpuri3.github.io after content or style edits. Use when working on files under content/, assets/, layouts/, static/, or hugo.toml and you need to (1) run a clean build, (2) render locally at localhost:1313, and (3) deploy to GitHub Pages via push.
---

# Website Release Loop

Use this workflow after every website change.

## Quick Commands

- Validate build: `bash skills/website-release-loop/scripts/release_loop.sh check`
- Preview locally: `bash skills/website-release-loop/scripts/release_loop.sh preview`
- Publish (commit + push): `bash skills/website-release-loop/scripts/release_loop.sh publish "your commit message"`

## Workflow

1. Run clean build.
2. Fix any build or rendering issues.
3. Start local server and review at `http://localhost:1313/`.
4. Push to `master` to trigger `.github/workflows/Deploy.yml`.

## Guardrails

- Keep commit scope intentional; do not blindly include unrelated files.
- Do not publish until user approves push.
- If the server is already running on `1313`, stop the old process before starting a new one.

## Notes

- Hosting is GitHub Pages via deployment workflow on push.
- If deployment status is needed, run `gh run list --workflow Deploy.yml --limit 1`.
