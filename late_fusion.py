import os
from argparse import ArgumentParser
import pandas as pd
import numpy as np

from config import TASKS, PREDICTION_FOLDER, HUMOR, PERCEPTION_LABELS, PERCEPTION, LOG_FOLDER
from main import get_eval_fn


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--task', type=str, required=True, choices=TASKS)
    parser.add_argument(
        '--label_dim',
        choices=PERCEPTION_LABELS,
        required=False,
        help='Specify the emotion dimension, only relevant for perception.')
    parser.add_argument(
        '--model_ids',
        nargs='+',
        required=True,
        help='model ids')
    parser.add_argument('--seeds', nargs='+', required=True, help='seeds')
    parser.add_argument('--result_csv', required=False, type=str)
    # add argument for for early fusion, mean, performance, max
    parser.add_argument(
        '--method', 
        type=str, 
        default='performance', 
        choices=['mean','max', 'performance', 'log']
    )

    args = parser.parse_args()
    # TODO add again
    #assert len(set(args.model_ids)) == len(args.model_ids), "Error, duplicate model file"
    # assert len(args.model_ids) >= 2, "For late fusion, please give at least 2 different models"

    if args.task == PERCEPTION:
        assert args.label_dim

    if args.seeds and len(args.seeds) == 1:
        args.seeds = [args.seeds[0]] * len(args.model_ids)
        assert len(args.model_ids) == len(args.seeds)

    if args.task == HUMOR:
        args.prediction_dirs = [
            os.path.join(
                PREDICTION_FOLDER,
                args.task,
                args.model_ids[i],
                args.seeds[i]) for i in range(len(args.model_ids))]
    elif args.task == PERCEPTION:
        args.prediction_dirs = [
            os.path.join(
                PREDICTION_FOLDER,
                args.task,
                args.label_dim,
                args.model_ids[i],
                args.seeds[i]) for i in range(len(args.model_ids)
            )
        ]
    if args.result_csv is not None:
        args.result_csv = os.path.join(
            LOG_FOLDER,
            'lf_metadata',
            args.task if args.task == HUMOR else f'{args.task}/{args.label_dim}',
            args.result_csv)
        os.makedirs(os.path.dirname(args.result_csv), exist_ok=True)
        if not args.result_csv.endswith('.csv'):
            args.result_csv += '.csv'
    return args


def create_humor_lf(df, weights=None):
    pred_arr = df[[c for c in df.columns if c.startswith(
        'prediction_')]].values
    if args.method == 'max':
        fused_preds = np.max(pred_arr, axis=1)
    elif args.method =='mean':
        fused_preds = np.mean(pred_arr, axis=1)
    else:
        if weights is None:
            # auto-compute weights based on performance
            labels = df['label'].values
            eval_fn, _ = get_eval_fn(HUMOR)
            weights = []
            for i in range(pred_arr.shape[1]):
                preds = pred_arr[:, i]
                weights.append(max(eval_fn(preds, labels) - 0.5, 0))
            print('Weights: ', weights)
            # weights = [1.] * pred_arr.shape[1]
            if all(w == 0 for w in weights):
                print('Only zeros')
                weights = [1/len(weights)] * len(weights)
        weights = np.array(weights) / np.sum(weights)
        for i, w in enumerate(weights.tolist()):
            preds = pred_arr[:, i]
            # normalise and weight
            preds = (preds - np.min(preds)) / (np.max(preds) - np.min(preds))
            preds = w * preds
            pred_arr[:, i] = preds
        fused_preds = np.sum(pred_arr, axis=1)

    if partition == 'devel':
        labels = df['label'].values
        return fused_preds, labels, weights
    else:
        return fused_preds, None, weights

def create_perception_lf(df, weights=None):
    pred_arr = df[[c for c in df.columns if c.startswith('prediction')]].values
    if args.method == 'max':
        fused_preds = np.max(pred_arr, axis=1)
    elif args.method == 'mean':
        fused_preds = np.mean(pred_arr, axis=1)
    elif args.method == 'log':
        fused_preds = np.log(np.sum(np.exp(pred_arr), axis=1))
    else: # performance
        if weights is None:
            # auto-compute weights based on performance
            labels = df['label'].values
            eval_fn,_ = get_eval_fn(PERCEPTION)
            weights = []
            for i in range(pred_arr.shape[1]):
                preds = pred_arr[:,i]
                # use all models for perception
                eval_per = eval_fn(preds, labels)
                weights.append(max(eval_per, 0))
            print('Weights: ', weights)
            #weights = [1.] * pred_arr.shape[1]
            if all(w==0 for w in weights):
                print('Only zeros')
                weights = [1/len(weights)] * len(weights)
        weights = np.array(weights) / np.sum(weights)
        for i, w in enumerate(weights.tolist()):
            preds = pred_arr[:, i]
            preds = (preds - np.min(preds)) / (np.max(preds) - np.min(preds))
            preds = w * preds
            pred_arr[:, i] = preds
        fused_preds = np.sum(pred_arr, axis=1)
    if partition == 'devel':
        labels = df['label'].values
        return fused_preds, labels, weights
    else:
        return fused_preds, None, weights


if __name__ == '__main__':
    args = parse_args()
    print(f"Task: {args.task}, method: {args.method}")
    ress = []       # hook for results
    weights = None  # gets set to devel weights at first call
    for partition in ['devel', 'test']:
        dfs = [
            pd.read_csv(
                os.path.join(
                    pred_dir,
                    f'predictions_{partition}.csv')) for pred_dir in args.prediction_dirs]
        # print loaded file
        print(f"Loaded {partition} data from {len(dfs)} models")
        meta_cols = [c for c in list(dfs[0].columns) if c.startswith('meta_')]
        print(f"Meta columns: {meta_cols}")
        for meta_col in meta_cols:
            assert all(np.all(df[meta_col].values == dfs[0]
                       [meta_col].values) for df in dfs)
        meta_df = dfs[0][meta_cols].copy()

        # for devel only
        if partition == 'devel':
            label_cols = [c for c in list(dfs[0].columns) if c.startswith('label')]
            print(f"Labels columns: {label_cols}")
            for label_col in label_cols:
                assert all(np.all(df[label_col].values == dfs[0][label_col].values) for df in dfs)
            label_df = dfs[0][label_cols].copy()

        prediction_dfs = []
        for i, df in enumerate(dfs):
            pred_df = df.drop(columns=meta_cols + label_cols)
            pred_df.rename(
                columns={
                    c: f'{c}_{args.model_ids[i]}' for c in pred_df.columns},
                inplace=True)
            prediction_dfs.append(pred_df)
        prediction_df = pd.concat(prediction_dfs, axis='columns')

        if partition == 'devel':
            full_df = pd.concat([meta_df, prediction_df, label_df], axis='columns')
        else:
            full_df = pd.concat([meta_df, prediction_df], axis='columns')
            
        if args.task == HUMOR:
            preds, labels, weights = create_humor_lf(full_df, weights=weights)
        elif args.task == PERCEPTION:
            preds, labels, weights = create_perception_lf(
                full_df, weights=weights)
            
        if not os.path.exists(os.path.join(PREDICTION_FOLDER, 
            'lf',                         
            args.task if args.task == HUMOR else f'{args.task}/{args.label_dim}')):
            os.makedirs(os.path.join(PREDICTION_FOLDER, 
                'lf', 
                    args.task if args.task == HUMOR else f'{args.task}/{args.label_dim}'),
                    exist_ok=True)
        
        # save fused prediction to new csv for devel set
        if partition == 'devel':
            new_df = full_df.copy()
            new_df['prediction'] = preds
            # crate lf_results folder if not exist
            new_df.to_csv(
                os.path.join(
                    PREDICTION_FOLDER,
                    'lf',
                    args.task if args.task == HUMOR else f'{args.task}/{args.label_dim}',
                    f'predictions_{partition}_lf.csv'),
                    index=False)
        # save fused prediction to new csv for test set
        if partition == 'test':
            new_df = full_df.copy()
            new_df['prediction'] = preds
            # crate lf_results folder if not exist
            new_df.to_csv(
                os.path.join(
                    PREDICTION_FOLDER,
                    'lf',
                    'humor' if args.task == HUMOR else f'{args.task}/{args.label_dim}',
                    f'predictions_{partition}_lf.csv'),
                    index=False)
        

        eval_fn, eval_str = get_eval_fn(args.task)

        if partition == 'devel':
            result = np.round(eval_fn(preds, labels), 4)
            print(f'{partition}: {result} {eval_str}')
            # devel: 0.2522 Pearson
            ress.append(result)

    if args.result_csv is not None:
        df = pd.DataFrame({
            'models': [args.model_ids],
            'weights': [str(weights)],
            'seeds': [args.seeds],
            'devel': [ress[0]],
            # 'test': [ress[1]]
        })
        if os.path.exists(args.result_csv):
            old_df = pd.read_csv(args.result_csv)
            df = pd.concat([old_df, df], axis='rows').reset_index(drop=True)
        df.to_csv(args.result_csv, index=False)
        # save metadata of prediction of test set to csv
        print(f'Saved to {args.result_csv}')
