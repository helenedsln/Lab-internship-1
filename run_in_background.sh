#!/bin/bash

# Run the Python script in the background
# nohup python3 -c "from german_generation_new_metrics import main_german_generate_final()" > output.log 2>&1 &

nohup python3 german_generation.py > output.log 2>&1 &
