import yaml
import json
import wandb
import random
import torch
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from pytorch_lightning.loggers import WandbLogger

from src import log, set_out, span
from src.utils import seed_everything
from src.data_loader import DataLoader
from src.preprocessor import Preprocessor
from src.dataset import RecTaskDataModule
from src.plmodel import RecTaskPLModel
from src.trainer import get_trainer
from src.submission import submission

import multiprocessing
from gensim.models import Word2Vec

rand = random.randint(0, 100000)

def run(config: dict) -> None:
    seed_everything(config["seed"], gpu_mode=True)

    with span("Load datasets"):
        train, test, sku_to_content = DataLoader(config, False).load_datasets()
        train = train[(train["product_sku_hash"].notnull()) & (train["product_action"] == "detail")]
        test = test[(test["product_sku_hash"].notnull()) & (test["product_action"] == "detail")]
        log(f"train: {train.shape}")
        log(f"test: {test.shape}")

    with span("Preprocess data"):
        train_session_ids = train["session_id_hash"].value_counts()
        test_session_ids = test["session_id_hash"].value_counts()
        train_session_ids = set(train_session_ids[train_session_ids > 1].index)
        test_session_ids = set(test_session_ids[test_session_ids > 1].index)

        train = train[train["session_id_hash"].isin(train_session_ids)]
        test = test[test["session_id_hash"].isin(test_session_ids)]

        train_sessions = train.groupby("session_id_hash")["product_sku_hash"].apply(lambda x: x.values).reset_index()
        test_sessions = test.groupby("session_id_hash")["product_sku_hash"].apply(lambda x: x.values).reset_index()
        sessions = pd.concat([train_sessions, test_sessions], sort=False)

    with span("Train W2V"):
        cores = multiprocessing.cpu_count()
        w2v_model = Word2Vec(
            min_count=1,
            window=5,
            vector_size=64,
            alpha=0.03,
            min_alpha=0.0007,
            workers=cores - 1
        )
        sentences = [list(ar) for ar in sessions["product_sku_hash"].to_list()]
        w2v_model.build_vocab(sentences)
        w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=50)

        output_path = Path(config["file_path"]["output_dir"]) / config["exp_name"]
        w2v_model.save(str(output_path / "word2vec.model"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
        config["exp_name"] = Path(f.name).stem

    output_path = Path(config["file_path"]["output_dir"]) / config["exp_name"]
    if not output_path.exists():
        output_path.mkdir(parents=True)
    set_out(output_path / "train_log.txt")

    log(f"configuration: {config}")
    run(config)
