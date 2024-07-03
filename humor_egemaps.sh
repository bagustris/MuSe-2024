#!/bin/bash

python3 main.py --predict --task humor --feature egemaps --normalize --use_gpu --model_dim 46 --rnn_n_layers 3 --d_fc_out 61 --rnn_dropout 0.15068854698759815 --linear_dropout 0.23652399437274957 --lr 2.064134198535456e-05 --batch_size 128 --loss bce --regularization 1.7232767056499437e-05 --rnn_type lstm --early_stopping_patience 24 --epochs 421 --activation prelu --residual --balance_humor
