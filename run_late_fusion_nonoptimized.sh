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

aggressive_model_ids=("
gru_2024-07-04-12-44_[ds]_[256_2_False_64]_[0.0005_256]
gru_2024-07-04-12-37_[egemaps]_[256_2_False_64]_[0.0005_256]
gru_2024-07-04-12-25_[w2v-msp]_[256_2_False_64]_[0.0005_256]
gru_2024-07-04-12-12_[vit-fer]_[256_2_False_64]_[0.0005_256]
gru_2024-07-04-12-05_[facenet512]_[256_2_False_64]_[0.0005_256]
gru_2024-07-04-11-58_[faus]_[256_2_False_64]_[0.0005_256]")

arrogant_model_ids=("
gru_2024-07-04-12-44_[ds]_[256_2_False_64]_[0.0005_256]
gru_2024-07-04-12-38_[egemaps]_[256_2_False_64]_[0.0005_256]
gru_2024-07-04-12-26_[w2v-msp]_[256_2_False_64]_[0.0005_256]
gru_2024-07-04-12-12_[vit-fer]_[256_2_False_64]_[0.0005_256]
gru_2024-07-04-12-05_[facenet512]_[256_2_False_64]_[0.0005_256]
gru_2024-07-04-11-59_[faus]_[256_2_False_64]_[0.0005_256]")

dominant_model_ids=("
gru_2024-07-04-12-45_[ds]_[256_2_False_64]_[0.0005_256]
gru_2024-07-04-12-38_[egemaps]_[256_2_False_64]_[0.0005_256]
gru_2024-07-04-12-27_[w2v-msp]_[256_2_False_64]_[0.0005_256]
gru_2024-07-04-12-13_[vit-fer]_[256_2_False_64]_[0.0005_256]
gru_2024-07-04-12-06_[facenet512]_[256_2_False_64]_[0.0005_256]
gru_2024-07-04-11-59_[faus]_[256_2_False_64]_[0.0005_256]")

enthusiastic_model_ids=("
gru_2024-07-04-12-46_[ds]_[256_2_False_64]_[0.0005_256]
gru_2024-07-04-12-39_[egemaps]_[256_2_False_64]_[0.0005_256]
gru_2024-07-04-12-28_[w2v-msp]_[256_2_False_64]_[0.0005_256]
gru_2024-07-04-12-13_[vit-fer]_[256_2_False_64]_[0.0005_256]
gru_2024-07-04-12-06_[facenet512]_[256_2_False_64]_[0.0005_256]
gru_2024-07-04-12-00_[faus]_[256_2_False_64]_[0.0005_256]")

friendly_model_ids=("
gru_2024-07-04-12-47_[ds]_[256_2_False_64]_[0.0005_256]
gru_2024-07-04-12-39_[egemaps]_[256_2_False_64]_[0.0005_256]
gru_2024-07-04-12-28_[w2v-msp]_[256_2_False_64]_[0.0005_256]
gru_2024-07-04-12-15_[vit-fer]_[256_2_False_64]_[0.0005_256]
gru_2024-07-04-12-07_[facenet512]_[256_2_False_64]_[0.0005_256]
gru_2024-07-04-12-00_[faus]_[256_2_False_64]_[0.0005_256]")

leader_like_model_ids=("
gru_2024-07-04-12-48_[ds]_[256_2_False_64]_[0.0005_256]
gru_2024-07-04-12-40_[egemaps]_[256_2_False_64]_[0.0005_256]
gru_2024-07-04-12-29_[w2v-msp]_[256_2_False_64]_[0.0005_256]
gru_2024-07-04-12-16_[vit-fer]_[256_2_False_64]_[0.0005_256]
gru_2024-07-04-12-07_[facenet512]_[256_2_False_64]_[0.0005_256]
gru_2024-07-04-12-01_[faus]_[256_2_False_64]_[0.0005_256]")


likeable_model_ids=("
gru_2024-07-04-12-48_[ds]_[256_2_False_64]_[0.0005_256]
gru_2024-07-04-12-40_[egemaps]_[256_2_False_64]_[0.0005_256]
gru_2024-07-04-12-30_[w2v-msp]_[256_2_False_64]_[0.0005_256]
gru_2024-07-04-12-17_[vit-fer]_[256_2_False_64]_[0.0005_256]
gru_2024-07-04-12-08_[facenet512]_[256_2_False_64]_[0.0005_256]
gru_2024-07-04-12-01_[faus]_[256_2_False_64]_[0.0005_256]")

assertiv_model_ids=("
gru_2024-07-04-12-49_[ds]_[256_2_False_64]_[0.0005_256]
gru_2024-07-04-12-40_[egemaps]_[256_2_False_64]_[0.0005_256]
gru_2024-07-04-12-31_[w2v-msp]_[256_2_False_64]_[0.0005_256]
gru_2024-07-04-12-18_[vit-fer]_[256_2_False_64]_[0.0005_256]
gru_2024-07-04-12-08_[facenet512]_[256_2_False_64]_[0.0005_256]
gru_2024-07-04-12-02_[faus]_[256_2_False_64]_[0.0005_256]")

confident_model_ids=("
gru_2024-07-04-12-50_[ds]_[256_2_False_64]_[0.0005_256]
gru_2024-07-04-12-41_[egemaps]_[256_2_False_64]_[0.0005_256]
gru_2024-07-04-12-31_[w2v-msp]_[256_2_False_64]_[0.0005_256]
gru_2024-07-04-12-18_[vit-fer]_[256_2_False_64]_[0.0005_256]
gru_2024-07-04-12-08_[facenet512]_[256_2_False_64]_[0.0005_256]
gru_2024-07-04-12-02_[faus]_[256_2_False_64]_[0.0005_256]")

independent_model_ids=("
gru_2024-07-04-12-51_[ds]_[256_2_False_64]_[0.0005_256]
gru_2024-07-04-12-41_[egemaps]_[256_2_False_64]_[0.0005_256]
gru_2024-07-04-12-32_[w2v-msp]_[256_2_False_64]_[0.0005_256]
gru_2024-07-04-12-20_[vit-fer]_[256_2_False_64]_[0.0005_256]
gru_2024-07-04-12-09_[facenet512]_[256_2_False_64]_[0.0005_256]
gru_2024-07-04-12-02_[faus]_[256_2_False_64]_[0.0005_256]")

risk_model_ids=("
gru_2024-07-04-12-52_[ds]_[256_2_False_64]_[0.0005_256]
gru_2024-07-04-12-41_[egemaps]_[256_2_False_64]_[0.0005_256]
gru_2024-07-04-12-32_[w2v-msp]_[256_2_False_64]_[0.0005_256]
gru_2024-07-04-12-21_[vit-fer]_[256_2_False_64]_[0.0005_256]
gru_2024-07-04-12-09_[facenet512]_[256_2_False_64]_[0.0005_256]
gru_2024-07-04-12-03_[faus]_[256_2_False_64]_[0.0005_256]")

sincere_model_ids=("
gru_2024-07-04-12-53_[ds]_[256_2_False_64]_[0.0005_256]
gru_2024-07-04-12-42_[egemaps]_[256_2_False_64]_[0.0005_256]
gru_2024-07-04-12-33_[w2v-msp]_[256_2_False_64]_[0.0005_256]
gru_2024-07-04-12-21_[vit-fer]_[256_2_False_64]_[0.0005_256]
gru_2024-07-04-12-10_[facenet512]_[256_2_False_64]_[0.0005_256]
gru_2024-07-04-12-03_[faus]_[256_2_False_64]_[0.0005_256]")


collaborative_model_ids=("
gru_2024-07-04-12-54_[ds]_[256_2_False_64]_[0.0005_256]
gru_2024-07-04-12-42_[egemaps]_[256_2_False_64]_[0.0005_256]
gru_2024-07-04-12-34_[w2v-msp]_[256_2_False_64]_[0.0005_256]
gru_2024-07-04-12-22_[vit-fer]_[256_2_False_64]_[0.0005_256]
gru_2024-07-04-12-10_[facenet512]_[256_2_False_64]_[0.0005_256]
gru_2024-07-04-12-03_[faus]_[256_2_False_64]_[0.0005_256]")

kind_model_ids=("
gru_2024-07-04-12-55_[ds]_[256_2_False_64]_[0.0005_256]
gru_2024-07-04-12-43_[egemaps]_[256_2_False_64]_[0.0005_256]
gru_2024-07-04-12-35_[w2v-msp]_[256_2_False_64]_[0.0005_256]
gru_2024-07-04-12-22_[vit-fer]_[256_2_False_64]_[0.0005_256]
gru_2024-07-04-12-10_[facenet512]_[256_2_False_64]_[0.0005_256]
gru_2024-07-04-12-04_[faus]_[256_2_False_64]_[0.0005_256]")

warm_model_ids=("
gru_2024-07-04-12-56_[ds]_[256_2_False_64]_[0.0005_256]
gru_2024-07-04-12-43_[egemaps]_[256_2_False_64]_[0.0005_256]
gru_2024-07-04-12-36_[w2v-msp]_[256_2_False_64]_[0.0005_256]
gru_2024-07-04-12-23_[vit-fer]_[256_2_False_64]_[0.0005_256]
gru_2024-07-04-12-11_[facenet512]_[256_2_False_64]_[0.0005_256]
gru_2024-07-04-12-04_[faus]_[256_2_False_64]_[0.0005_256]")

good_natured_model_ids=("
gru_2024-07-04-12-56_[ds]_[256_2_False_64]_[0.0005_256]
gru_2024-07-04-12-43_[egemaps]_[256_2_False_64]_[0.0005_256]
gru_2024-07-04-12-36_[w2v-msp]_[256_2_False_64]_[0.0005_256]
gru_2024-07-04-12-24_[vit-fer]_[256_2_False_64]_[0.0005_256]
gru_2024-07-04-12-11_[facenet512]_[256_2_False_64]_[0.0005_256]
gru_2024-07-04-12-04_[faus]_[256_2_False_64]_[0.0005_256]")

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
