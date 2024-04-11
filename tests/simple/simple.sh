#!/bin/bash

# Check if the user provided an argument
if [ $# -eq 0 ]; then
    echo "Usage: $0 <input>"
    exit 1
fi

# Echo the input argument
echo "The input argument is: $1"

cp "$2/input.txt" "$2/output.txt"

echo "$1" > "$2/output.txt"

echo "Text has been written to output.txt"