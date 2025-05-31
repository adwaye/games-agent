#!/bin/bash


SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

python \
$SCRIPT_DIR'/src/games_agent/scripts/run_train.py' \
-cp $SCRIPT_DIR'src/games_agent/configs/' \
-m agent.env.reward_function='merge_only,'full_merge','log_full_merge' \
agent.env.reward_scaling='total_value','total_steps,'current'