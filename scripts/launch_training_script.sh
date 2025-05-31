#!/bin/bash

python \
/home/adwaye/AI_Models/games_agent/src/games_agent/scripts/run_train.py \
-cp /home/adwaye/AI_Models/games_agent/src/games_agent/configs/ \
-m agent.env.reward_function='merge_only,'full_merge','log_full_merge' \
agent.env.reward_scaling='total_value','total_steps,'current'