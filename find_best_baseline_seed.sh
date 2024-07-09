#!/bin/bash

# Define the models and seeds
models=("ds" "egemaps" "facenet512" "faus" "vit" "w2v-msp")
seeds=(101 102 103 104 105)

# Define the tasks and labels
tasks=("perception")
perception_labels=("aggressive" "arrogant" "dominant" "enthusiastic" "friendly" "leader_like" "likeable" "assertiv" "confident" "independent" "risk" "sincere" "collaborative" "kind" "warm" "good_natured")

# Function to run the late fusion script and extract the score
run_and_get_score() {
    local task=$1
    local label=$2
    local model=$3
    local seed=$4
    
    # Run the late fusion script
    output=$(python late_fusion_baseline.py --task "$task" --label_dim "$label" --model_ids "$model" --seeds "$seed" --method performance)
    
    # Extract the score from the output
    score=$(echo "$output" | grep "devel:" | awk '{print $2}')
    
    echo "$score"
}

# Main loop
for task in "${tasks[@]}"; do
    if [ "$task" == "perception" ]; then
        for label in "${perception_labels[@]}"; do
            echo "Task: $task, Label: $label"
            for model in "${models[@]}"; do
                best_score=0
                best_seed=""
                for seed in "${seeds[@]}"; do
                    score=$(run_and_get_score "$task" "$label" "$model" "$seed")
                    if (( $(echo "$score > $best_score" | bc -l) )); then
                        best_score=$score
                        best_seed=$seed
                    fi
                done
                echo "  Model: $model, Best Seed: $best_seed, Best Score: $best_score"
            done
            echo ""
        done
    else
        echo "Task: $task"
        for model in "${models[@]}"; do
            best_score=0
            best_seed=""
            for seed in "${seeds[@]}"; do
                score=$(run_and_get_score "$task" "" "$model" "$seed")
                if (( $(echo "$score > $best_score" | bc -l) )); then
                    best_score=$score
                    best_seed=$seed
                fi
            done
            echo "  Model: $model, Best Seed: $best_seed, Best Score: $best_score"
        done
        echo ""
    fi
done
