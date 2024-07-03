#!/bin/bash

# new model, best: 0.67
python3 main.py --predict --task humor --feature w2v-msp --use_gpu --model_dim 108 --rnn_n_layers 3 --rnn_bi --d_fc_out 66 --rnn_dropout 0.2441539220241588 --linear_dropout 0.33889685898641164 --lr 1.5439258192962333e-05 --batch_size 256 --loss bce --regularization 0.0018944535957281512 --rnn_type rnn --early_stopping_patience 17 --epochs 217 --activation elu



# old model, best: 0.62
# --model_dim 81 --rnn_n_layers 2 --d_fc_out 70 --rnn_dropout 0.4199386523418765 --linear_dropout 0.23374369134365058 --lr 1.1367222248713414e-05 --batch_size 256 --loss bce --regularization 0.007759313346449349 --rnn_type gru --early_stopping_patience 28 --epochs 814 --activation prelu
