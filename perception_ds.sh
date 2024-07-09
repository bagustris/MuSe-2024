labels=('aggressive' 'arrogant' 'dominant' 'enthusiastic' 'friendly' 'leader_like' 'likeable' 'assertiv' 'confident' 'independent' 'risk' 'sincere' 'collaborative' 'kind' 'warm' 'good_natured')

# run for all labels
for label in "${labels[@]}"; do

    python3 main.py --predict --task perception --use_gpu --feature ds --model_dim 99 --rnn_n_layers 3 --d_fc_out 51 --rnn_dropout 0.10655567959259438 --linear_dropout 0.47148537180840366 --lr 0.000595530307014647 --batch_size 128 --loss mse --regularization 0.00015464702104317393 --rnn_type gru --early_stopping_patience 10 --epochs 355 --activation mish --label_dim $label --data_augmentation

done
