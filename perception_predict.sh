!#/usr/bin/bash

labels=('aggressive' 'arrogant' 'dominant' 'enthusiastic' 'friendly' 'leader_like' 'likeable' 'assertiv' 'confident' 'independent' 'risk' 'sincere' 'collaborative' 'kind' 'warm' 'good_natured')

# run for all labels
for label in "${labels[@]}"; do
    python3 main.py \
    --task perception \
    --use_gpu \
    --feature vit-fer \
    --model_dim 82 \
    --label_dim "$label" \
    --rnn_n_layers 2 \
    --rnn_bi \
    --d_fc_out 107 \
    --rnn_dropout 0.15526697083996654 \
    --linear_dropout 0.25346360181954036 \
    --lr 2.1752251822197732e-05 \
    --batch_size 64 \
    --loss pcc \
    --regularization 0.003074309607843819
done

# python3 main.py --predict --task perception --use_gpu --feature vit-fer --seed 110 --n_seed 1 --model_dim 82 --rnn_n_layers 2 --rnn_bi --d_fc_out 107 --rnn_dropout 0.15526697083996654 --linear_dropout 0.25346360181954036 --lr 2.1752251822197732e-05 --batch_size 64 --loss pcc --regularization 0.003074309607843819