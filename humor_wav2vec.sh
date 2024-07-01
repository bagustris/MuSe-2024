#!/bin/bash

python3 main.py --predict --task humor --feature w2v-msp --optuna --use_gpu --model_dim 81 --rnn_n_layers 2 --d_fc_out 70 --rnn_dropout 0.4199386523418765 --linear_dropout 0.23374369134365058 --lr 1.1367222248713414e-05 --batch_size 256 --loss bce --regularization 0.007759313346449349 --rnn_type gru --early_stopping_patience 28 --epochs 814 --activation prelu
