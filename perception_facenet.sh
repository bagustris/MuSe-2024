#!/bin/bash

labels=('aggressive' 'arrogant' 'dominant' 'enthusiastic' 'friendly' 'leader_like' 'likeable' 'assertiv' 'confident' 'independent' 'risk' 'sincere' 'collaborative' 'kind' 'warm' 'good_natured')

# run for all labels
for label in "${labels[@]}"; do

    python3 main.py --predict --task perception --use_gpu --feature facenet512 --model_dim 121 --rnn_n_layers 2 --rnn_bi --d_fc_out 51 --rnn_dropout 0.2847632754771246 --linear_dropout 0.38572690452967634 --lr 0.00029315203097385074 --batch_size 256 --loss ccc --regularization 9.998358683184358e-05 --rnn_type lstm --early_stopping_patience 14 --epochs 396 --activation mish --residual --label_dim $label --data_augmentation
    
done
