#!/bin/bash

python3 main.py --predict --task humor --feature vit-fer --use_gpu --model_dim 60 --rnn_n_layers 3 --d_fc_out 72 --rnn_dropout 0.2898883137364525 --linear_dropout 0.3023313599445294 --lr 1.4960184172944572e-05 --batch_size 64 --loss bce --regularization 0.00012578117170170507 --rnn_type gru --early_stopping_patience 9 --epochs 740 --activation prelu
