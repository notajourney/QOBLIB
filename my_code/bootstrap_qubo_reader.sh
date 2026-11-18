#!/usr/bin/env bash
set -euo pipefail

# Run from any working directory.
# Requires Python 3.12 and its venv module.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3.12}"

echo "===== Python environment ====="
"$PYTHON_BIN" -m venv .venv
.venv/bin/python -m pip install -r my_code/qubo_reader/requirements.txt

echo "===== Extract QOBLib QUBOs ====="
.venv/bin/python my_code/extract_qubos.py --all

echo "===== Build inventory ====="
.venv/bin/python -m my_code.qubo_reader.inventory --all

echo "===== Reader tests ====="
.venv/bin/python -m unittest discover \
    -s tests -p 'test_qubo_*.py' -v

echo "===== Dataset count ====="
find extracted_qb -type f -name '*.qs' | wc -l

echo "===== Bootstrap completed ====="
