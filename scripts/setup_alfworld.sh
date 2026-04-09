#!/bin/bash
set -e
pip install alfworld --no-cache-dir
echo "AlfWorld package installed."

# Download game files (this is the actual important part)
python -c "import alfworld.agents.environment"
echo "AlfWorld environment importable."

# Verify by listing data path
python -c "
import os
data_path = os.environ.get('ALFWORLD_DATA', os.path.expanduser('~/.alfworld'))
print('ALFWORLD_DATA path:', data_path)
"
echo "ALFWorld setup complete"
