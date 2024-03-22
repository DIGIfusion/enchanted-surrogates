#!/bin/bash

# Check if the user provided an argument
if [ $# -eq 0 ]; then
    echo "Usage: $0 <input>"
    exit 1
fi

# Echo the input argument
echo "The input argument is: $1"
