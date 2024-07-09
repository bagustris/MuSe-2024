#/usr/bin/bash

# python code to evalute baseline models given in PTH file

labels=('aggressive' 'arrogant' 'dominant' 'enthusiastic' 'friendly' 'leader_like' 'likeable' 'assertiv' 'confident' 'independent' 'risk' 'sincere' 'collaborative' 'kind' 'warm' 'good_natured')
features=('faus' 'facenet512' 'vit-fer' 'w2v-msp' 'egemaps --normalize' 'ds')
seeds=(101 102 103 104 105)

for label in "${labels[@]}"; do

    for feature in "${features[@]}"; do
    echo "Evaluating feature $feature for label $label"
    feature_dir=$feature
    # if feature == vif-fet change to vit
    if [[ "$feature" = "vit-fer" ]]; then
        feature_dir="vit"
    # if feature == egemaps --nnormalize, feature_dir=egemaps
    elif [[ "$feature" = "egemaps --normalize" ]]; then
        feature_dir="egemaps"
    fi
        for seed in "${seeds[@]}"; do
            python main.py --task perception --feature $feature --eval_model /home/bagus/data/MuSe2024/perception_models/perception/$label/$feature_dir --eval_seed $seed --use_gpu --predict
        done
        echo "--------------seed finish--------------"
    done
    echo "--------------feature finish--------------"
done