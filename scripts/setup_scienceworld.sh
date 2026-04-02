#!/bin/bash
set -e
pip install scienceworld
python -c "
import scienceworld
env = scienceworld.ScienceWorldEnv()
tasks = env.getTaskNames()
print(f'ScienceWorld: {len(tasks)} tasks available')
env.close()
"
echo "ScienceWorld setup complete"
