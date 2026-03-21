# Week 2 — Git & GitHub Summary

## What I Learned

### The Mental Model
Git is a timeline of snapshots. Every commit is a permanent snapshot
you can return to at any time.

### The Three Zones
1. **Working Directory** — where you edit files
2. **Staging Area** — files queued for the next snapshot (git add)
3. **Repository** — permanent snapshots (git commit)

### Key Commands
- `git init` — start tracking a folder
- `git add .` — stage everything
- `git commit -m "msg"` — take the snapshot
- `git push` — send to GitHub
- `git checkout -b name` — create and switch branch
- `git merge name` — merge branch into current
- `git pull` — get latest from GitHub

### .gitignore
- Protects secrets (.env) from ever reaching GitHub
- Excludes generated files (outputs/, __pycache__/)
- Patterns: wildcards (*.pyc), folders (.venv/), negation (!)

### Pull Requests
- PRs = proposed merges with discussion and review
- Always branch → push → PR → review → merge
- Never push directly to main in a team setting

## Repos Live on GitHub
- agentic-ai-roadmap: github.com/Aniruddhahrid/agentic-ai-roadmap
- react-learning: github.com/Aniruddhahrid/react-learning