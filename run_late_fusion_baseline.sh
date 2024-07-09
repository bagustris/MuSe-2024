#!/bin/bash

features=("ds" "egemaps" "facenet512" "faus" "vit" "w2v-msp")

labels=(
    'aggressive' 
    'arrogant'
    'assertiv' 
    'confident'
    'dominant'
    'independent' 
    'risk'
    'leader_like'
    'collaborative' 
    'enthusiastic' 
    'friendly'  
    'good_natured'
    'kind' 
    'likeable'   
    'sincere' 
    'warm' 
)

declare -A seeds
seeds[aggressive]="103 102 103 105 104 101"
seeds[arrogant]="103 102 103 101 102 104"
seeds[assertiv]="103 103 101 104 104 105"
seeds[confident]="103 103 101 104 104 105"
seeds[dominant]="103 103 105 101 103 101"
seeds[independent]="103 103 101 104 103 105"
seeds[risk]="103 103 105 103 103 103"
seeds[leader_like]="103 103 101 104 104 105"
seeds[collaborative]="103 103 101 105 103 105"
seeds[enthusiastic]="103 103 101 104 104 105"
seeds[friendly]="103 103 101 101 105 105"
seeds[good_natured]="103 103 104 101 102 105"
seeds[kind]="103 103 104 105 101 105"
seeds[likeable]="103 103 101 101 101 105"
seeds[sincere]="103 103 101 105 103 105"
seeds[warm]="103 103 104 101 101 105"

# loop over labels
for label in "${labels[@]}"; do
    # Get the seeds for the current label
    label_seeds=${seeds[$label]}
    
    echo "Label: $label"
    # Run the late fusion
    python3 late_fusion_baseline.py --task perception --label_dim "$label" --model_ids "${features[@]}" --seeds $label_seeds #--method max
    echo "-------------------------" 
done
