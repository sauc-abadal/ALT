#!/bin/bash

#SBATCH --time=1:00:00
#SBATCH --mem-per-cpu=20G

# Function to concatenate JSON files
concatenate_files_without_newline() {
    local output_file="$1"
    shift
    local input_files=("$@")
    
    # Concatenate files with newline character in between
    for file in "${input_files[@]}"; do
        cat "$file"
    done > "$output_file"
    
    echo "Concatenated JSON files into $output_file"
}

# Check if at least two arguments are provided
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <output_file> <input_file1> [<input_file2> ...]"
    exit 1
fi

# Extract output file name
output_file="$1"
shift

# Concatenate JSON files
concatenate_files_without_newline "$output_file" "$@"

