# Git upload instructions

The release folder is initialized on branch `main` but is intentionally not committed: author identity, repository URL, visibility, and license are owner decisions.

```powershell
git status --short
git add .
git commit -m "Release six-family PIGNO portfolio campaign"
git remote add origin https://github.com/OWNER/REPOSITORY.git
git push -u origin main
```

Before the commit:

1. choose and add a license;
2. run `python scripts/verify_repository.py`;
3. inspect `git status --short`;
4. publish external binaries in a durable data archive and add their URLs to `data/external/EXTERNAL_BINARY_MANIFEST.csv`.

Do not add raw `.mph` files or the multi-gigabyte HDF5 artifacts directly to ordinary Git history.
