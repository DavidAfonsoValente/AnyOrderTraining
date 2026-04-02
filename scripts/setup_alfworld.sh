#!/bin/bash
set -e
pip install alfworld
python -c "import alfworld; print('alfworld version:', alfworld.__version__)"
# Download game files
python -c "import alfworld.agents.environment; alfworld.agents.environment.AlfredTWEnv"
# Verify by listing available configs
python -c "
import alfworld
import os
data_path = os.environ.get('ALFWORLD_DATA', os.path.expanduser('~/.alfworld'))
print('ALFWORLD_DATA:', data_path)
print('Exists:', os.path.exists(data_path))
"
echo "ALFWorld setup complete"
