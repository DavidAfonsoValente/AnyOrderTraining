#!/bin/bash
set -e
pip install -r requirements.txt
bash scripts/setup_alfworld.sh
bash scripts/setup_scienceworld.sh
bash scripts/setup_webshop.sh
python scripts/download_data.py
echo "Full setup complete"
