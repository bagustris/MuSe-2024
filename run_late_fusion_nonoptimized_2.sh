#!/usr/bin/bash
# bash code for running non-optimized perception (perception_full.sh)

# usage:./run_late_fusion_nonoptimized.sh argument
# argument:
#   a: run all labels and late fusion
#   lf: run late fusion for a specific label
#   label: label to run late fusion for

# get arguments
# show help is no arguments are passed
if [ $# -eq 0 ]; then
    echo "Usage:./run_late_fusion_nonoptimized.sh argument"
    echo "argument:"
    echo "  -all: run all labels and late fusion"
    echo "  -lf: run late fusion for a specific label"
    echo "  -label: label to run late fusion for"
    exit 1
fi

# find last modified directory in aggresive
aggressive_model_ids=("gru_2024-07-08-18-46_[ds]_[256_2_False_64]_[0.0005_512]
gru_2024-07-08-18-39_[egemaps]_[256_2_False_64]_[0.0005_512]
gru_2024-07-08-18-25_[w2v-msp]_[256_2_False_64]_[0.0005_512]
gru_2024-07-08-18-12_[vit-fer]_[256_2_False_64]_[0.0005_512]
gru_2024-07-08-18-04_[facenet512]_[256_2_False_64]_[0.0005_512]
gru_2024-07-08-17-57_[faus]_[256_2_False_64]_[0.0005_512]
")



arrogant_model_ids=("
gru_2024-07-08-18-47_[ds]_[256_2_False_64]_[0.0005_512]
gru_2024-07-08-18-39_[egemaps]_[256_2_False_64]_[0.0005_512]
gru_2024-07-08-18-26_[w2v-msp]_[256_2_False_64]_[0.0005_512]
gru_2024-07-08-18-13_[vit-fer]_[256_2_False_64]_[0.0005_512]
gru_2024-07-08-18-05_[facenet512]_[256_2_False_64]_[0.0005_512]
gru_2024-07-08-17-58_[faus]_[256_2_False_64]_[0.0005_512]")

dominant_model_ids=("
gru_2024-07-08-18-47_[ds]_[256_2_False_64]_[0.0005_512]
gru_2024-07-08-18-39_[egemaps]_[256_2_False_64]_[0.0005_512]
gru_2024-07-08-18-27_[w2v-msp]_[256_2_False_64]_[0.0005_512]
gru_2024-07-08-18-14_[vit-fer]_[256_2_False_64]_[0.0005_512]
gru_2024-07-08-18-05_[facenet512]_[256_2_False_64]_[0.0005_512]
gru_2024-07-08-17-59_[faus]_[256_2_False_64]_[0.0005_512]")

enthusiastic_model_ids=("
gru_2024-07-08-18-48_[ds]_[256_2_False_64]_[0.0005_512]
gru_2024-07-08-18-40_[egemaps]_[256_2_False_64]_[0.0005_512]
gru_2024-07-08-18-28_[w2v-msp]_[256_2_False_64]_[0.0005_512]
gru_2024-07-08-18-14_[vit-fer]_[256_2_False_64]_[0.0005_512]
gru_2024-07-08-18-06_[facenet512]_[256_2_False_64]_[0.0005_512]
gru_2024-07-08-17-59_[faus]_[256_2_False_64]_[0.0005_512]")

friendly_model_ids=("
gru_2024-07-08-18-49_[ds]_[256_2_False_64]_[0.0005_512]
gru_2024-07-08-18-41_[egemaps]_[256_2_False_64]_[0.0005_512]
gru_2024-07-08-18-28_[w2v-msp]_[256_2_False_64]_[0.0005_512]
gru_2024-07-08-18-16_[vit-fer]_[256_2_False_64]_[0.0005_512]
gru_2024-07-08-18-06_[facenet512]_[256_2_False_64]_[0.0005_512]
gru_2024-07-08-18-00_[faus]_[256_2_False_64]_[0.0005_512]")

leader_like_model_ids=("
gru_2024-07-08-18-50_[ds]_[256_2_False_64]_[0.0005_512]
gru_2024-07-08-18-41_[egemaps]_[256_2_False_64]_[0.0005_512]
gru_2024-07-08-18-29_[w2v-msp]_[256_2_False_64]_[0.0005_512]
gru_2024-07-08-18-16_[vit-fer]_[256_2_False_64]_[0.0005_512]
gru_2024-07-08-18-07_[facenet512]_[256_2_False_64]_[0.0005_512]
gru_2024-07-08-18-00_[faus]_[256_2_False_64]_[0.0005_512]")


likeable_model_ids=("
gru_2024-07-08-18-51_[ds]_[256_2_False_64]_[0.0005_512]
gru_2024-07-08-18-42_[egemaps]_[256_2_False_64]_[0.0005_512]
gru_2024-07-08-18-30_[w2v-msp]_[256_2_False_64]_[0.0005_512]
gru_2024-07-08-18-18_[vit-fer]_[256_2_False_64]_[0.0005_512]
gru_2024-07-08-18-07_[facenet512]_[256_2_False_64]_[0.0005_512]
gru_2024-07-08-18-00_[faus]_[256_2_False_64]_[0.0005_512]")

assertiv_model_ids=("
gru_2024-07-08-18-52_[ds]_[256_2_False_64]_[0.0005_512]
gru_2024-07-08-18-42_[egemaps]_[256_2_False_64]_[0.0005_512]
gru_2024-07-08-18-31_[w2v-msp]_[256_2_False_64]_[0.0005_512]
gru_2024-07-08-18-19_[vit-fer]_[256_2_False_64]_[0.0005_512]
gru_2024-07-08-18-08_[facenet512]_[256_2_False_64]_[0.0005_512]
gru_2024-07-08-18-01_[faus]_[256_2_False_64]_[0.0005_512]")

confident_model_ids=("
gru_2024-07-08-18-53_[ds]_[256_2_False_64]_[0.0005_512]
gru_2024-07-08-18-42_[egemaps]_[256_2_False_64]_[0.0005_512]
gru_2024-07-08-18-32_[w2v-msp]_[256_2_False_64]_[0.0005_512]
gru_2024-07-08-18-19_[vit-fer]_[256_2_False_64]_[0.0005_512]
gru_2024-07-08-18-08_[facenet512]_[256_2_False_64]_[0.0005_512]
gru_2024-07-08-18-01_[faus]_[256_2_False_64]_[0.0005_512]")

independent_model_ids=("
gru_2024-07-08-18-54_[ds]_[256_2_False_64]_[0.0005_512]
gru_2024-07-08-18-43_[egemaps]_[256_2_False_64]_[0.0005_512]
gru_2024-07-08-18-32_[w2v-msp]_[256_2_False_64]_[0.0005_512]
gru_2024-07-08-18-21_[vit-fer]_[256_2_False_64]_[0.0005_512]
gru_2024-07-08-18-09_[facenet512]_[256_2_False_64]_[0.0005_512]
gru_2024-07-08-18-02_[faus]_[256_2_False_64]_[0.0005_512]")

risk_model_ids=("
gru_2024-07-08-18-55_[ds]_[256_2_False_64]_[0.0005_512]
gru_2024-07-08-18-43_[egemaps]_[256_2_False_64]_[0.0005_512]
gru_2024-07-08-18-33_[w2v-msp]_[256_2_False_64]_[0.0005_512]
gru_2024-07-08-18-21_[vit-fer]_[256_2_False_64]_[0.0005_512]
gru_2024-07-08-18-09_[facenet512]_[256_2_False_64]_[0.0005_512]
gru_2024-07-08-18-02_[faus]_[256_2_False_64]_[0.0005_512]")

sincere_model_ids=("gru_2024-07-08-18-56_[ds]_[256_2_False_64]_[0.0005_512]
gru_2024-07-08-18-44_[egemaps]_[256_2_False_64]_[0.0005_512]
gru_2024-07-08-18-33_[w2v-msp]_[256_2_False_64]_[0.0005_512]
gru_2024-07-08-18-22_[vit-fer]_[256_2_False_64]_[0.0005_512]
gru_2024-07-08-18-09_[facenet512]_[256_2_False_64]_[0.0005_512]
gru_2024-07-08-18-03_[faus]_[256_2_False_64]_[0.0005_512]")


collaborative_model_ids=("
gru_2024-07-08-18-57_[ds]_[256_2_False_64]_[0.0005_512]
gru_2024-07-08-18-44_[egemaps]_[256_2_False_64]_[0.0005_512]
gru_2024-07-08-18-34_[w2v-msp]_[256_2_False_64]_[0.0005_512]
gru_2024-07-08-18-23_[vit-fer]_[256_2_False_64]_[0.0005_512]
gru_2024-07-08-18-10_[facenet512]_[256_2_False_64]_[0.0005_512]
gru_2024-07-08-18-03_[faus]_[256_2_False_64]_[0.0005_512]")

kind_model_ids=("
gru_2024-07-08-18-58_[ds]_[256_2_False_64]_[0.0005_512]
gru_2024-07-08-18-45_[egemaps]_[256_2_False_64]_[0.0005_512]
gru_2024-07-08-18-35_[w2v-msp]_[256_2_False_64]_[0.0005_512]
gru_2024-07-08-18-23_[vit-fer]_[256_2_False_64]_[0.0005_512]
gru_2024-07-08-18-10_[facenet512]_[256_2_False_64]_[0.0005_512]
gru_2024-07-08-18-03_[faus]_[256_2_False_64]_[0.0005_512]")

warm_model_ids=("gru_2024-07-08-18-59_[ds]_[256_2_False_64]_[0.0005_512]
gru_2024-07-08-18-45_[egemaps]_[256_2_False_64]_[0.0005_512]
gru_2024-07-08-18-36_[w2v-msp]_[256_2_False_64]_[0.0005_512]
gru_2024-07-08-18-24_[vit-fer]_[256_2_False_64]_[0.0005_512]
gru_2024-07-08-18-11_[facenet512]_[256_2_False_64]_[0.0005_512]
gru_2024-07-08-18-03_[faus]_[256_2_False_64]_[0.0005_512]")

good_natured_model_ids=("gru_2024-07-08-19-00_[ds]_[256_2_False_64]_[0.0005_512]
gru_2024-07-08-18-45_[egemaps]_[256_2_False_64]_[0.0005_512]
gru_2024-07-08-18-37_[w2v-msp]_[256_2_False_64]_[0.0005_512]
gru_2024-07-08-18-24_[vit-fer]_[256_2_False_64]_[0.0005_512]
gru_2024-07-08-18-11_[facenet512]_[256_2_False_64]_[0.0005_512]
gru_2024-07-08-18-04_[faus]_[256_2_False_64]_[0.0005_512]")

#!/usr/bin/bash


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


if [ "$1" = "-label" ] || [ "$1" = "-all" ]; then
    for label in "${labels[@]}"; do
        # Construct the variable name
        model_ids_var="${label}_model_ids"
        
        echo "Processing label: $label"
        
        # Iterate over each model ID for the current label
        for model_id in ${!model_ids_var}; do
            echo "Evaluating model: $model_id"
            
            # Run late_fusion.py for each individual model
            python late_fusion.py --task perception --label_dim "$label" --model_ids "$model_id" --seeds 101
        done
        
        echo "-------------------------"
    done
fi


# if argument is -lf or -all
if [ "$1" = "-lf" ] || [ "$1" = "-all" ]; then
    for label in "${labels[@]}"; do
        # Get the variable name for the model_ids
        model_ids_var="${label}_model_ids"
        
        # Print information: label and model ids
        echo processing late fusion for "$label" with "$label"_model_ids
        
        # Use the specific model_ids for each label, change method here
        python late_fusion.py --task perception --label_dim "$label" --model_ids ${!model_ids_var} --seeds 101

    done
fi
