#!/bin/bash

# Create plots directory if it doesn't exist
mkdir -p ./plots

# Process all .npy files in the cache directory
for mask_file in cache/*.npy; do
    # Skip if no .npy files found
    [ -e "$mask_file" ] || continue
    
    # Extract filename without path and extension
    filename=$(basename "$mask_file")
    
    # Skip pvalues files
    if [[ $filename == *"pvalues"* ]]; then
        continue
    fi
    
    # Extract parameters from filename using pattern matching
    model_name=$(echo $filename | cut -d'_' -f1)
    network=$(echo $filename | grep -o "network=[^_]*" | cut -d'=' -f2)
    foundation=$(echo $filename | grep -o "foundation=[^_]*" | cut -d'=' -f2)
    pooling=$(echo $filename | grep -o "pooling=[^_]*" | cut -d'=' -f2)
    percentage=$(echo $filename | grep -o "perc=[^_]*" | cut -d'=' -f2)
    
    echo "Processing $filename"
    echo "Model: $model_name"
    echo "Network: $network"
    echo "Foundation: $foundation"
    echo "Pooling: $pooling"
    echo "Percentage: $percentage"
    echo "----------------------------------------"
    
    python -m analyze_mask \
        --mask-path "$mask_file" \
        --model-name "$model_name" \
        --percentage "$percentage" \
        --pooling "$pooling" \
        --foundation "$foundation"
done