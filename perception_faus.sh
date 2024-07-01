#!/usr/bin/bash

labels=('aggressive' 'arrogant' 'dominant' 'enthusiastic' 'friendly' 'leader_like' 'likeable' 'assertiv' 'confident' 'independent' 'risk' 'sincere' 'collaborative' 'kind' 'warm' 'good_natured')

# run for all labels
for label in "${labels[@]}"; do

python3 main.py --predict --task perception --use_gpu --feature faus --model_dim 102 --rnn_n_layers 1 --d_fc_out 83 --rnn_dropout 0.4576787280124962 --linear_dropout 0.29425340368911657 --lr 0.0009465919851445551 --batch_size 256 --loss mse --regularization 0.0001775504728846824 --rnn_type gru --early_stopping_patience 24 --epochs 339 --activation prelu --residual --label_dim $label

done
