#!/bin/bash


python3 main.py --predict --task humor --feature faus --use_gpu --model_dim 41 --rnn_n_layers 4 --d_fc_out 50 --rnn_dropout 0.281515963666792 --linear_dropout 0.39250650021067673 --lr 1.0506570488504824e-05 --batch_size 128 --loss bce --regularization 0.0005691968518754981 --rnn_type gru --early_stopping_patience 29 --epochs 368 --activation elu --residual --balance_humor
