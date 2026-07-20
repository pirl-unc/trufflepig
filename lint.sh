#!/bin/bash
set -o errexit

python -m ruff check trufflepig tests
echo 'Passes ruff check'
