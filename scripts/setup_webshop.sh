#!/bin/bash
set -e
mkdir -p third_party
if [ ! -d third_party/WebShop ]; then
  git clone https://github.com/princeton-nlp/WebShop third_party/WebShop
fi
cd third_party/WebShop
pip install -e .
# Download product data
python setup.py download_data 2>/dev/null || bash setup.sh
# Verify server can start
python -c "from web_agent_site.app import app; print('WebShop importable')"
echo "WebShop setup complete"
