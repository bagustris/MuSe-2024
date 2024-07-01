#!/usr/bin/bash

labels=('aggressive' 'arrogant' 'dominant' 'enthusiastic' 'friendly' 'leader_like' 'likeable' 'assertiv' 'confident' 'independent' 'risk' 'sincere' 'collaborative' 'kind' 'warm' 'good_natured')

# run for all labels
for label in "${labels[@]}"; do

   python3 main.py --predict --task perception --use_gpu --feature egemaps --normalize --model_dim 125 --rnn_n_layers 1 --rnn_bi --d_fc_out 32 --rnn_dropout 0.43781061724432796 --linear_dropout 0.4080124158830384 --lr 0.0008314184545955257 --batch_size 64 --loss mse --regularization 0.007172438873077759 --rnn_type gru --label_dim $label
    
done
