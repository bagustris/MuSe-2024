#!/usr/bin/bash

features=('faus' 'facenet512' 'vit-fer' 'w2v-msp' 'egemaps --normalize' 'ds')
for feature in "${features[@]}"; do
    python3 main.py --predict --task perception --use_gpu --feature $feature --optuna
done