#!/bin/bash

aggressive_model_ids=("
gru_2024-06-28-19-32_[egemaps]_[125_1_True_32]_[0.0008314184545955257_64]
gru_2024-06-28-19-33_[faus]_[102_1_False_83]_[0.0009465919851445551_256]
gru_2024-06-28-19-39_[ds]_[99_3_False_51]_[0.000595530307014647_128]
lstm_2024-06-28-19-15_[vit-fer]_[78_1_False_65]_[0.00032916684358508247_256]
lstm_2024-06-28-19-36_[facenet512]_[121_2_True_51]_[0.00029315203097385074_256]
lstm_2024-06-28-19-38_[w2v-msp]_[67_4_False_37]_[0.0004987280253678834_256]")

arrogant_model_ids=("gru_2024-06-28-19-32_[egemaps]_[125_1_True_32]_[0.0008314184545955257_64] gru_2024-06-28-19-37_[faus]_[102_1_False_83]_[0.0009465919851445551_256] gru_2024-06-28-19-41_[ds]_[99_3_False_51]_[0.000595530307014647_128] lstm_2024-06-28-19-16_[vit-fer]_[78_1_False_65]_[0.00032916684358508247_256] lstm_2024-06-28-19-37_[facenet512]_[121_2_True_51]_[0.00029315203097385074_256] lstm_2024-06-28-19-40_[w2v-msp]_[67_4_False_37]_[0.0004987280253678834_256]")

# labels=('aggressive' 'arrogant' 'dominant' 'enthusiastic' 'friendly' 'leader_like' 'likeable' 'assertiv' 'confident' 'independent' 'risk' 'sincere' 'collaborative' 'kind' 'warm' 'good_natured')

labels=('aggressive' 'arrogant')

for label in "${labels[@]}"; do
    # Get the variable name for the model_ids
    model_ids_var="${label}_model_ids"
    
    # Print information: label and model ids
    echo processing late fusion for $label
    
    # Use the specific model_ids for each label
    python late_fusion.py --task perception --label_dim "$label" --model_ids ${!model_ids_var} --seeds 107
done

