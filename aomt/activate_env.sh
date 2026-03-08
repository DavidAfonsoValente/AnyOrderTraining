#!/bin/bash
# This script loads the correct python module, sets memory limits, and activates the virtual environment.
echo 'Attempting to load Python 3.11 module...'
module load python/3.11 || true
echo 'Setting memory limit to unlimited...'
ulimit -m unlimited
echo 'Activating uv environment at /home/davidvalente/AnyOrderTraining/aomt/dFactory/VeOmni/.venv...'
source /home/davidvalente/AnyOrderTraining/aomt/dFactory/VeOmni/.venv/bin/activate
export PYTHONPATH=/home/davidvalente/AnyOrderTraining:${PYTHONPATH}
echo 'Environment activated.'
