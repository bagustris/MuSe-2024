# Evaluation of RNN Models for MuSe-2024


[Homepage](https://www.muse-challenge.org) || [Baseline Paper](https://www.researchgate.net/publication/380664467_The_MuSe_2024_Multimodal_Sentiment_Analysis_Challenge_Social_Perception_and_Humor_Recognition)


## Sub-challenges and Results 
For details, please see the [Baseline Paper](https://www.researchgate.net/publication/380664467_The_MuSe_2024_Multimodal_Sentiment_Analysis_Challenge_Social_Perception_and_Humor_Recognition). If you want to sign up for the challenge, please fill out the form 
[here](https://www.muse-challenge.org/challenge/participate).

* MuSe-Perception: predicting 16 different dimensions of social perception (e.g. Assertiveness, Likability, Warmth,...). 
 *Official baseline*: **.3573** mean Pearson's correlation over all 16 classes.

* MuSe-Humor: predicting the presence/absence of humor in cross-cultural (German/English) football press conference recordings. 
*Official baseline*: **.8682** AUC.


## Installation
It is highly recommended to run everything in a Python virtual environment. Please make sure to install the packages listed 
in ``requirements.txt`` and adjust the paths in `config.py` (especially ``BASE_PATH`` and ``HUMOR_PATH`` and/or ``PERCEPTION_PATH``, respectively). 

You can then, e.g., run the unimodal baseline reproduction calls in the ``*_bagus.sh`` file provided for each sub-challenge.  

```bash
$ ./perception_bagus.sh
```

## Settings
The ``main.py`` script is used for training and evaluating models.  Most important options:
* ``--task``: choose either `perception` or `humor` 
* ``--feature``: choose a feature set provided in the data (in the ``PATH_TO_FEATURES`` defined in ``config.py``). Adding 
``--normalize`` ensures normalization of features (recommended for ``eGeMAPS`` features).
* Options defining the model architecture: ``d_rnn``, ``rnn_n_layers``, ``rnn_bi``, ``d_fc_out``
* Options for the training process: ``--epochs``, ``--lr``, ``--seed``,  ``--n_seeds``, ``--early_stopping_patience``,
``--reduce_lr_patience``,   ``--rnn_dropout``, ``--linear_dropout``
* In order to use a GPU, please add the flag ``--use_gpu``
* Predict labels for the test set: ``--predict``
* Specific parameter for MuSe-Perception: ``label_dim`` (one of the 16 labels, cf. ``config.py``), ``win_len`` and ``hop_len`` for segmentation.

For more details, please see the ``parse_args()`` method in ``main.py``.

## Reproducing the baselines 
Please note that exact reproducibility can not be expected due to dependence on hardware. 
### Unimodal models
For every challenge, a ``*_full.sh`` file is provided with the respective call (and, thus, configuration) for each of the precomputed features.
Moreover, you can directly load one of the checkpoints corresponding to the results in the baseline paper. Note that 
the checkpoints are only available to registered participants. 


### Prediction
To predict test files, one can use the ``--predict`` option in ``main.py``. This will create prediction folders under the folder specified as the prediction directory in ``config.py``.

```
python3 main.py --predict --task perception --use_gpu --feature vit-fer
```

### Checkpoint model
A checkpoint model can be loaded and evaluated as follows:

``` 
main.py --task humor --feature faus --eval_model /your/checkpoint/directory/humor_faus/model_102.pth
``` 

### Evaluate single model
To evaluate single model, we can use `late_fusion.py` with a model for model_ids. For instance, we have a list of single model in variable `models`.

```bash
 for model in ${models[@]}; do python late_fusion.py --task perception --label_dim dominant --model_ids $model --seeds 107; done
 ```
 The `models` variable can be obtained from ouput directory and list them (using `ls`) and then pasted to terminal as follows:

 ```bash
 $ models=('gru_2024-06-28-20-00_[egemaps]_[125_1_True_32]_[0.0008314184545955257_64]'
> 'gru_2024-06-28-20-06_[faus]_[102_1_False_83]_[0.0009465919851445551_256]'
> 'gru_2024-06-28-20-09_[ds]_[99_3_False_51]_[0.000595530307014647_128]'
> 'lstm_2024-06-28-19-35_[vit-fer]_[78_1_False_65]_[0.00032916684358508247_256]'
> 'lstm_2024-06-28-19-59_[facenet512]_[121_2_True_51]_[0.00029315203097385074_256]'
> 'lstm_2024-06-28-20-02_[w2v-msp]_[67_4_False_37]_[0.0004987280253678834_256]'
> )
```

### Late Fusion
We utilize a simple late fusion approach, which averages different models' predictions. 
First, predictions for development and test set have to be created using the ``--predict`` option in ``main.py``. 
This will create prediction folders under the folder specified as the prediction directory in ``config.py``.

Then, ``late_fusion.py`` merges these predictions:
* ``--task``: choose either `humor` or `perception` 
* ``--label_dim``: for MuSe-Perception, cf. ``PERCEPTION_LABELS`` in ``config.py``
* ``--model_ids``: list of model IDs, whose predictions are to be merged. These predictions must first be created (``--predict`` in ``main.py`` or ``personalisation.py``). 
  The `model_id` is a folder under the ``{config.PREDICTION_DIR}/humor`` for humor and ``{config.PREDICTION_FOLDER}/perception/{label_dim}`` for perception. 
  It is the parent folder of the folders named after the seeds (e.g. ``101``). These contain the files ``predictions_devel.csv`` and ``predictoins_test.csv``
* ``--seeds``: seeds for the respective model IDs.  

Example:  
```
$ python late_fusion.py --task perception --label_dim assertiv --model_ids RNN_2024-06-10-10-30_[vit-fer]_[64_1_False_64]_[0.0001_256] RNN_2024-06-10-10-33_[vit-fer]_[64_1_False_64]_[0.0001_256] --seeds 101
```

### Hyperparameter Tuning

```
python main.py --task perception --feature vit-fer --optuna
```


##  Citation:

```bibtex
@inproceedings{10.1145/3689062.3689082,
author = {Atmaja, Bagus Tris},
title = {Feature-wise Optimization and Performance-weighted Multimodal Fusion for Social Perception Recognition},
year = {2024},
isbn = {9798400711992},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3689062.3689082},
doi = {10.1145/3689062.3689082},
abstract = {Automatic social perception recognition is a new task to mimic the measurement of human traits, which was previously done by humans via questionnaires. We evaluated unimodal and multimodal systems to predict agentive and communal traits from the LMU-ELP dataset. We optimized variants of recurrent neural networks from each feature from audio and video data and then fused them to predict the traits. Results on the development set show a consistent trend that multimodal fusion outperforms unimodal systems. The performance-weighted fusion also consistently outperforms mean and maximum fusions. We found two important factors that influence the performance of performance-weighted fusion. These factors are normalization and the number of models.},
booktitle = {Proceedings of the 5th on Multimodal Sentiment Analysis Challenge and Workshop: Social Perception and Humor},
pages = {28–35},
numpages = {8},
keywords = {multimodal fusion, parameter optimization, sentiment analysis, social perception},
location = {Melbourne VIC, Australia},
series = {MuSe'24}
}
```
