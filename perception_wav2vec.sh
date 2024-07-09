#!/bin/bash

labels=('aggressive' 'arrogant' 'dominant' 'enthusiastic' 'friendly' 'leader_like' 'likeable' 'assertiv' 'confident' 'independent' 'risk' 'sincere' 'collaborative' 'kind' 'warm' 'good_natured')

# run for all labels
for label in "${labels[@]}"; do

    python3 main.py --predict --task perception --use_gpu --feature w2v-msp --model_dim 67 --rnn_n_layers 4 --d_fc_out 37 --rnn_dropout 0.22968230271659562 --linear_dropout 0.40283229860576847 --lr 0.0004987280253678834 --batch_size 256 --loss mse --regularization 1.283779594433699e-05 --rnn_type lstm --early_stopping_patience 6 --epochs 386 --activation leakyrelu --residual --label_dim $label --data_augmentation

done
