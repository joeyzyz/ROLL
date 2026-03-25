#!/bin/bash
set +x

ROLL_PATH="/zhouyu/ROLL"
CONFIG_PATH="agentic_demo"
# CONFIG_PATH=$(basename $(dirname $0))
export PYTHONPATH="$ROLL_PATH:$PYTHONPATH"
python examples/start_agentic_pipeline.py --config_path $CONFIG_PATH  --config_name agent_val_frozen_lake_single_node_demo
