import argparse
import os
import random
import sys
from datetime import datetime

import numpy
import torch
from dateutil import tz
from torch import nn

import config
from config import TASKS, PERCEPTION, HUMOR
from data_parser import load_data
from dataset import MuSeDataset, custom_collate_fn
from eval import evaluate, calc_auc, calc_pearsons
from model import Model
from train import train_model
from utils import Logger, seed_worker, log_results

import optuna
from optuna.samplers import TPESampler

# import logging


def parse_args():
    parser = argparse.ArgumentParser(description="MuSe 2024.")

    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=TASKS,
        help=f"Specify the task from {TASKS}.",
    )
    parser.add_argument(
        "--feature", required=True, help="Specify the features used (only one)."
    )
    parser.add_argument(
        "--label_dim", default="assertiv", choices=config.PERCEPTION_LABELS
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Specify whether to normalize features (default: False).",
    )
    parser.add_argument(
        "--model_dim",
        type=int,
        default=64,
        help="Specify the number of hidden states in the RNN (default: 64).",
    )
    parser.add_argument(
        "--rnn_n_layers",
        type=int,
        default=1,
        help="Specify the number of layers for the RNN (default: 1).",
    )
    parser.add_argument(
        "--rnn_bi",
        action="store_true",
        help="Specify whether the RNN is bidirectional or not (default: False).",
    )
    parser.add_argument(
        "--d_fc_out",
        type=int,
        default=64,
        help="Specify the number of hidden neurons in the output layer (default: 64).",
    )
    parser.add_argument("--rnn_dropout", type=float, default=0.2)
    parser.add_argument("--linear_dropout", type=float, default=0.5)
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Specify the number of epochs (default: 100).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Specify the batch size (default: 256).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0001,
        help="Specify initial learning rate (default: 0.0001).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=101,
        help="Specify the initial random seed (default: 101).",
    )
    parser.add_argument(
        "--n_seeds",
        type=int,
        default=5,
        help="Specify number of random seeds to try (default: 5).",
    )
    parser.add_argument(
        "--result_csv",
        default=None,
        help="Append the results to this csv (or create it, if it "
        "does not exist yet). Incompatible with --predict",
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=15,
        help="Patience for early stopping",
    )
    parser.add_argument(
        "--use_gpu",
        action="store_true",
        help="Specify whether to use gpu for training (default: False).",
    )
    parser.add_argument(
        "--cache",
        action="store_true",
        help="Specify whether to cache data as pickle file (default: False).",
    )
    parser.add_argument(
        "--predict",
        action="store_true",
        help="Specify when no test labels are available; "
             "test predictions will be saved "
             "(default: False). Incompatible with result_csv",
    )
    parser.add_argument(
        "--regularization", type=float, required=False, default=0.0, help="L2-Penalty"
    )
    # evaluation only arguments
    parser.add_argument(
        "--eval_model",
        type=str,
        default=None,
        help="Specify model which is to be evaluated; "
             "no training with this option (default: False).",
    )
    parser.add_argument(
        "--eval_seed",
        type=str,
        default=None,
        help="Specify seed to be evaluated; "
             "only considered when --eval_model is given.",
    )
    # add argument for loss function, default to mse, choices: mae, ccc, pcc
    parser.add_argument(
        "--loss",
        type=str,
        default="mse" if PERCEPTION else "bce",
        choices=["mse", "mae", "ccc", "pcc", "bce"],
        help="Specify the loss function to be used (default: mse) for perception task.",
    )

    # add argument for rnn_type, default to gru, choices: lstm, rnn
    parser.add_argument(
        "--rnn_type",
        type=str,
        default="gru",
        choices=["lstm", "gru", "rnn"],
        help="Specify the RNN type to be used (default: gru).",
    )

    # argument for activation function
    parser.add_argument(
        "--activation",
        type=str,
        default="relu",
        choices=["relu", "gelu", "elu", "leakyrelu", "prelu", "rrelu", "mish"],
        help="Specify the activation function to be used in the output layer (default: relu).",
    )

    # add residual argument
    parser.add_argument(
        "--residual",
        action="store_true",
        help="Specify whether to use residual connections in the RNN (default: False).",
    )

    # add balancing argument for humor
    parser.add_argument(
        "--balance_humor",
        action="store_true",
        help="Specify whether to balance humor data (default: False). Only works for humor task.",
    )

    # use optuna to tune hyperparameters above
    parser.add_argument(
        "--optuna",
        action="store_true",
        help="Specify whether to use optuna to tune hyperparameters for perception (default: False).",
    )

    args = parser.parse_args()
    if args.result_csv is not None and args.predict:
        print("--result_csv is not compatible with --predict")
        sys.exit(-1)
    elif args.task == "perception" and args.balance_humor:
        print("--balance_humor is not compatible with perception task")
        sys.exit(-1)

    if args.eval_model:
        assert args.eval_seed
    return args


def pcc_loss(preds, labels):
    """Pearson correlation coefficient loss """
    var_preds = preds - torch.mean(preds)
    var_labels = labels - torch.mean(labels)
    pcc = torch.sum(var_preds * var_labels) / (torch.sqrt(torch.sum(var_preds ** 2))
                                               * torch.norm(torch.sum(var_labels ** 2)))
    return 1 - pcc


def ccc_loss(preds, labels):
    """Concordance correlation coefficient loss"""
    mean_preds = torch.mean(preds)
    mean_labels = torch.mean(labels)
    cov = torch.mean((preds - mean_preds) * (labels - mean_labels))
    var_preds = torch.var(preds)
    var_labels = torch.var(labels)
    return 1 - 2 * cov / (var_preds + var_labels + (mean_preds - mean_labels) ** 2)


def get_loss_fn(task):
    if task == HUMOR:
        # https://github.com/NVIDIA/pix2pixHD/issues/9
        return nn.BCEWithLogitsLoss(), "Binary Crossentropy"
    elif task == PERCEPTION:
        if args.loss == "mse":
            return nn.MSELoss(reduction="mean"), "MSE"
        elif args.loss == "mae":
            return nn.L1Loss(reduction="mean"), "L1"
        elif args.loss == "pcc":
            return pcc_loss, "PCC"
        elif args.loss == "ccc":
            return ccc_loss, "CCC"
        else:
            raise ValueError("Unknown loss function")


def get_eval_fn(task):
    if task == PERCEPTION:
        return calc_pearsons, "Pearson"
    elif task == HUMOR:
        return calc_auc, "AUC"


def objective(trial):
    args.model_dim = trial.suggest_int("model_dim", 32, 128)
    args.rnn_n_layers = trial.suggest_int("rnn_n_layers", 1, 4)
    args.rnn_bi = trial.suggest_categorical("rnn_bi", [True, False])
    args.d_fc_out = trial.suggest_int("d_fc_out", 32, 128)
    args.rnn_dropout = trial.suggest_float("rnn_dropout", 0.1, 0.5)
    args.linear_dropout = trial.suggest_float("linear_dropout", 0.1, 0.5)
    args.lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    args.batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
    args.loss = trial.suggest_categorical(
        "loss", ["mse", "mae", "ccc", "pcc"] if args.task == PERCEPTION else ["bce"])
    args.regularization = trial.suggest_float(
        "regularization", 1e-5, 1e-2, log=True)
    args.rnn_type = trial.suggest_categorical(
        "rnn_type", ["lstm", "gru", "rnn"])
    args.early_stopping_patience = trial.suggest_int(
        "early_stopping_patience", 5, 30)
    args.epochs = trial.suggest_int("epochs", 10, 1000)
    args.activation = trial.suggest_categorical(
        "activation", ["relu", "gelu", "elu", "leakyrelu", "prelu", "rrelu", "mish"])
    args.residual = trial.suggest_categorical("residual", [True, False])
    args.balance_humor = trial.suggest_categorical(
        "balance_humor", [True, False])

    # Load data, create datasets, define loss and evaluation functions
    data = load_data(
        args.task,
        args.paths,
        args.feature,
        args.label_dim,
        args.normalize,
        save=args.cache,
        balance_humor=args.balance_humor,
    )
    datasets = {partition: MuSeDataset(data, partition)
                for partition in data.keys()}

    loss_fn, loss_str = get_loss_fn(args.task)
    eval_fn, eval_str = get_eval_fn(args.task)

    if args.task == PERCEPTION:
        val_scores = []
        for label_dim in config.PERCEPTION_LABELS:
            args.label_dim = label_dim

            data_loader = {}
            for partition, dataset in datasets.items():
                batch_size = args.batch_size if partition == "train" else 2 * args.batch_size
                shuffle = True if partition == "train" else False
                data_loader[partition] = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    num_workers=4,
                    worker_init_fn=seed_worker,
                    collate_fn=custom_collate_fn,
                )

            model = Model(args)

            val_loss, val_score, _ = train_model(
                args.task,
                model,
                data_loader,
                args.epochs,
                args.lr,
                args.paths["model"],
                args.seed,
                use_gpu=args.use_gpu,
                loss_fn=loss_fn,
                eval_fn=eval_fn,
                eval_metric_str=eval_str,
                regularization=args.regularization,
                early_stopping_patience=args.early_stopping_patience,
            )

            val_scores.append(val_score)

        avg_val_score = numpy.mean(val_scores)
        return avg_val_score

    else:  # HUMOR task
        data_loader = {}
        for partition, dataset in datasets.items():
            batch_size = args.batch_size if partition == "train" else 2 * args.batch_size
            shuffle = True if partition == "train" else False
            data_loader[partition] = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=4,
                worker_init_fn=seed_worker,
                collate_fn=custom_collate_fn,
            )

        model = Model(args)

        val_loss, val_score, _ = train_model(
            args.task,
            model,
            data_loader,
            args.epochs,
            args.lr,
            args.paths["model"],
            args.seed,
            use_gpu=args.use_gpu,
            loss_fn=loss_fn,
            eval_fn=eval_fn,
            eval_metric_str=eval_str,
            regularization=args.regularization,
            early_stopping_patience=args.early_stopping_patience,
        )

        return val_score


def main(args):
    # ensure reproducibility
    numpy.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # label_dim is only for perception
    args.label_dim = args.label_dim if args.task == PERCEPTION else ""
    print("Loading data ...")
    args.paths["partition"] = os.path.join(
        config.PATH_TO_METADATA[args.task], "partition.csv"
    )

    data = load_data(
        args.task,
        args.paths,
        args.feature,
        args.label_dim,
        args.normalize,
        save=args.cache,
    )
    datasets = {partition: MuSeDataset(data, partition)
                for partition in data.keys()}

    args.d_in = datasets["train"].get_feature_dim()

    args.n_targets = config.NUM_TARGETS[args.task]
    args.n_to_1 = args.task in config.N_TO_1_TASKS

    loss_fn, loss_str = get_loss_fn(args.task)
    eval_fn, eval_str = get_eval_fn(args.task)

    if args.eval_model is None:  # Train and validate for each seed
        seeds = range(args.seed, args.seed + args.n_seeds)
        val_losses, val_scores, best_model_files, test_scores = [], [], [], []

        for seed in seeds:
            torch.manual_seed(seed)
            data_loader = {}
            for (
                partition,
                dataset,
            ) in datasets.items():  # one DataLoader for each partition
                batch_size = (
                    args.batch_size if partition == "train" else 2 * args.batch_size
                )
                # shuffle only for train partition
                shuffle = True if partition == "train" else False
                data_loader[partition] = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    num_workers=4,
                    worker_init_fn=seed_worker,
                    collate_fn=custom_collate_fn,
                )

            model = Model(args)

            print("=" * 50)
            print(
                f"Training model... [seed {seed}] for at most {args.epochs} epochs")

            val_loss, val_score, best_model_file = train_model(
                args.task,
                model,
                data_loader,
                args.epochs,
                args.lr,
                args.paths["model"],
                seed,
                use_gpu=args.use_gpu,
                loss_fn=loss_fn,
                eval_fn=eval_fn,
                eval_metric_str=eval_str,
                regularization=args.regularization,
                early_stopping_patience=args.early_stopping_patience,
            )
            # restore best model encountered during training
            model = torch.load(best_model_file)

            # run evaluation only if test labels are available
            if not args.predict:
                test_loss, test_score = evaluate(
                    args.task,
                    model,
                    data_loader["test"],
                    loss_fn=loss_fn,
                    eval_fn=eval_fn,
                    use_gpu=args.use_gpu,
                )
                test_scores.append(test_score)
                print(f"[Test {eval_str}]:  {test_score:7.4f}")

            val_losses.append(val_loss)
            val_scores.append(val_score)

            best_model_files.append(best_model_file)

        # find best performing seed
        best_idx = val_scores.index(max(val_scores))

        print("=" * 50)
        print(
            f'Best {eval_str} on [Val] for seed {seeds[best_idx]}: '
            f'[Val {eval_str}]: {val_scores[best_idx]:7.4f}'
            f"{f' | [Test {eval_str}]: {test_scores[best_idx]:7.4f}' if not args.predict else ''}"
        )
        print("=" * 50)

        # best model of all of the seeds
        model_file = best_model_files[best_idx]
        if args.result_csv is not None:
            log_results(
                args.result_csv,
                params=args,
                seeds=list(seeds),
                metric_name=eval_str,
                model_files=best_model_files,
                test_results=test_scores,
                val_results=val_scores,
                best_idx=best_idx,
            )

    else:  # Evaluate existing model (No training)
        model_file = os.path.join(
            args.paths["model"], f"model_{args.eval_seed}.pth")
        model = torch.load(
            model_file,
            map_location=torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("cpu"),
        )
        data_loader = {}
        for partition, dataset in datasets.items():  # one DataLoader for each partition
            batch_size = (
                args.batch_size if partition == "train" else 2 * args.batch_size
            )
            # shuffle only for train partition
            shuffle = True if partition == "train" else False
            data_loader[partition] = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=4,
                worker_init_fn=seed_worker,
                collate_fn=custom_collate_fn,
            )
        _, valid_score = evaluate(
            args.task,
            model,
            data_loader["devel"],
            loss_fn=loss_fn,
            eval_fn=eval_fn,
            use_gpu=args.use_gpu,
        )
        print(f"Evaluating {model_file}:")
        print(f"[Val {eval_str}]: {valid_score:7.4f}")
        if not args.predict:
            _, test_score = evaluate(
                args.task,
                model,
                data_loader["test"],
                loss_fn=loss_fn,
                eval_fn=eval_fn,
                use_gpu=args.use_gpu,
            )
            print(f"[Test {eval_str}]: {test_score:7.4f}")

    # Make predictions for the test partition;
    # this option is set if there are no test labels
    if args.predict:
        print("Predicting devel and test samples...")
        best_model = torch.load(model_file, map_location=config.device)
        # predict all
        for split in ["train", "devel", "test"]:
            evaluate(
                args.task,
                best_model,
                data_loader[split],
                loss_fn=loss_fn,
                eval_fn=eval_fn,
                use_gpu=args.use_gpu,
                predict=True,
                prediction_path=args.paths["predict"],
                filename=f"predictions_{split}.csv",
            )

        print(f'Find predictions in {os.path.join(args.paths["predict"])}'
        )

    if args.optuna:
        study = optuna.create_study(
            direction="maximize",
            storage=f"sqlite:///{os.path.join('logs', f'{args.task}_{args.feature}_optuna.db')}",
            sampler=TPESampler())
        # Adjust the number of trials as needed
        study.optimize(objective, n_trials=100)

        # print best value and best params
        print(f"Best value: {study.best_value}")
        print(f"Best hyperparameters:{study.best_params}")

        # Set the best hyperparameters to args
        for key, value in study.best_params.items():
            setattr(args, key, value)

        # save best hyperparameters to a file for each feature in logs/optuna
        current_date = datetime.now().strftime("%Y-%m-%d")
        with open(os.path.join("logs/optuna/",
                  f"{args.task}_{args.feature}_{current_date}_best_params.txt"), "w") as f:
            f.write(str(study.best_params))
    print("Done.")


if __name__ == "__main__":
    print("Start", flush=True)
    args = parse_args()

    args.log_file_name = "{}_{}_[{}]_[{}_{}_{}_{}]_[{}_{}]".format(
        args.rnn_type,
        datetime.now(tz=tz.gettz()).strftime("%Y-%m-%d-%H-%M"),
        args.feature.replace(os.path.sep, "-"),
        args.model_dim,
        args.rnn_n_layers,
        args.rnn_bi,
        args.d_fc_out,
        args.lr,
        args.batch_size,
    )

    # adjust your paths in config.py
    task_id = (
        args.task
        if args.task != PERCEPTION
        else os.path.join(args.task, args.label_dim)
    )
    args.paths = {
        "log": os.path.join(config.LOG_FOLDER, task_id)
        if not args.predict
        else os.path.join(config.LOG_FOLDER, task_id, "prediction"),
        "data": os.path.join(config.DATA_FOLDER, task_id),
        "model": os.path.join(
            config.MODEL_FOLDER,
            task_id,
            args.log_file_name if not args.eval_model else args.eval_model,
        ),
    }
    if args.predict:
        if args.eval_model:
            args.paths["predict"] = os.path.join(
                config.PREDICTION_FOLDER, task_id, args.eval_model, args.eval_seed
            )
        else:
            args.paths["predict"] = os.path.join(
                config.PREDICTION_FOLDER, task_id, args.log_file_name
            )

    for folder in args.paths.values():
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)

    args.paths.update(
        {
            "features": config.PATH_TO_FEATURES[args.task],
            "labels": config.PATH_TO_LABELS[args.task],
            "partition": config.PARTITION_FILES[args.task],
        }
    )

    sys.stdout = Logger(os.path.join(
        args.paths["log"], args.log_file_name + ".txt"))
    print(" ".join(sys.argv))

    # set timer
    start = datetime.now()
    print(f"Start time: {start.strftime('%Y-%m-%d %H:%M:%S')}")

    main(args)

    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Elapsed time: {datetime.now() - start}")
    print("DONE.", flush=True)

    # os.system(f"rm -r {config.OUTPUT_PATH}")
