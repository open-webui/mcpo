# Safe Git Routine (Beginner-Friendly)

This routine avoids pushing directly to `main` and keeps your fork clean.

## One-time setup

```powershell
git remote -v
```

If `upstream` is missing, add it (replace URL with the original repo):

```powershell
git remote add upstream https://github.com/<ORIGINAL_OWNER>/mcpo.git
```

Optional safety alias (push current branch only):

```powershell
git config --global alias.pushsafe "push origin HEAD"
```

## Daily workflow

1. Start clean from `main`:

```powershell
git checkout main
git fetch upstream
git merge --ff-only upstream/main
git push origin main
```

2. Create a feature branch (never code on `main`):

```powershell
git checkout -b feat/<short-name>
```

3. Work, commit, and push:

```powershell
git add -A
git commit -m "feat: <what changed>"
git push -u origin HEAD
```

4. Open a PR in GitHub from your feature branch into your target branch.

## If you already have local work on a branch (like `mcppo`)

```powershell
git checkout mcppo
git add -A
git commit -m "wip: checkpoint"
git push -u origin mcppo
```

## Quick safety checks

```powershell
git branch --show-current
git status --short --branch
```

If current branch is `main`, stop before commit/push unless you are intentionally syncing `main`.

## Note for this repo

`.github/workflows/publish.yaml` publishes on pushes to `main` and release events.
So keep regular feature work off `main` to avoid accidental release-related automation.
