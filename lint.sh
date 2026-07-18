#!/bin/bash
set -o errexit

ruff check trufflepig tests
echo 'Passes ruff check'
