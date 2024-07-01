#!/bin/bash

python3 main.py --predict --task humor --feature facenet512 --optuna --use_gpu --model_dim 88 --rnn_n_layers 2 --d_fc_out 43 --rnn_dropout 0.28076794880465966 --linear_dropout 0.42174401265880956 --lr 1.0300864576609383e-05 --batch_size 256 --loss bce --regularization 0.003926482657436495 --rnn_type rnn --early_stopping_patience 23 --epochs 213 --activation rrelu --balance_humor
