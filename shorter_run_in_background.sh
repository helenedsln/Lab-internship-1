#!/bin/bash

# Run the Python script in the background
# nohup python3 -c "from shorter_german_generation import main_german_generate_shorter()" > output.log 2>&1 &

nohup python3 shorter_german_generation.py > shorter_output.log 2>&1 &