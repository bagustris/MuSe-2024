#!/bin/bash

python3 main.py --predict --task humor --feature ds --use_gpu --model_dim 32 --rnn_n_layers 4 --d_fc_out 126 --rnn_dropout 0.47967344587947075 --linear_dropout 0.16667118075616363 --lr 0.00042779102647868677 --batch_size 256 --loss bce --regularization 0.005155637487837036 --rnn_type lstm --early_stopping_patience 29 --epochs 649 --activation gelu --residual
