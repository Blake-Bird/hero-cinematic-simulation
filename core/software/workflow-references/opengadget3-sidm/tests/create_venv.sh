#!/bin/bash

python3 -m venv OG3Venv
source OG3Venv/bin/activate
pip install --upgrade pip
pip install -r tests/Python-requirements.txt
