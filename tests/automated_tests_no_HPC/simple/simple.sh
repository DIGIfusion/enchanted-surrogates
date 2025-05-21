#!/bin/bash

# Check if the user provided an argument
if [ $# -eq 0 ]; then
    echo "Usage: $0 <input>"
    exit 1
fi

# Echo the input argument
echo "The input argument is: $1"
echo "The run_dir is: $2"
echo "ls of run_dir"
ls $2
cp "$2/in.json" "$2/out.json"
