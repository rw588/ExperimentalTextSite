# Citation Style

This directory holds the CSL citation style file.

Download the American Physical Society style from:
https://www.zotero.org/styles/american-physics-society

```bash
curl -o american-physics-society.csl \
  "https://www.zotero.org/styles/american-physics-society"
```

The CI workflow downloads it automatically. For local builds, run the command above once.

The `.csl` file itself is not tracked in git (see `.gitignore`) because it is a third-party
file that should be fetched from its authoritative source.
