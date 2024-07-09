

!#/usr/bin/bash

labels=('aggressive' 'arrogant' 'dominant' 'enthusiastic' 'friendly' 'leader_like' 'likeable' 'assertiv' 'confident' 'independent' 'risk' 'sincere' 'collaborative' 'kind' 'warm' 'good_natured')

# run for all labels
for label in "${labels[@]}"; do

    python3 main.py --predict --task perception --use_gpu --feature vit-fer --model_dim 78 --rnn_n_layers 1 --d_fc_out 65 --rnn_dropout 0.30830354542651284 --linear_dropout 0.4168513790131495 --lr 0.00032916684358508247 --batch_size 256 --loss pcc --regularization 0.001307092046278891 --rnn_type lstm --label_dim $label --data_augmentation
    
done
