#!/bin/bash

# List of features
features=(ds egemaps facenet faus vitfer wav2vec)

# Loop through each feature
for feature in "${features[@]}"
do
    # Construct the filename
    filename="perception_${feature}.sh"
    
    # Check if the file exists
    if [ -f "$filename" ]; then
        echo "Running $filename"
        # Run the bash script
        bash "$filename"
    else
        echo "Warning: $filename does not exist. Skipping."
    fi
done

echo "All perception scripts have been executed."