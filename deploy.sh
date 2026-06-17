#!/usr/bin/env bash
# Release trufflepig to PyPI (pirl-trufflepig) and tag the version.
# Matches the deploy.sh convention across the sibling repos (pirlygenes, isovar, …).
# Always deploy via this script — never a manual twine invocation.
set -euo pipefail

./test.sh
python3 -m pip install --upgrade build
python3 -m pip install --upgrade twine
rm -rf dist
python3 -m build
python3 -m twine upload dist/*
git tag "$(python3 trufflepig/version.py)"
git push --tags
