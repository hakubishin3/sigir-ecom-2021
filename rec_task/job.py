import yaml
import argparse
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold

from src import log, set_out, span
from src.utils import seed_everything
from src.data_loader import DataLoader
from src.preprocessor import Preprocessor
from src.dataset import RecTaskDataModule


def run(config: dict, debug: bool, holdout: bool) -> None:
    with span("Load datasets"):
        train, test, sku_to_content = DataLoader(config, debug).load_datasets()
        test_session_ids = set(test["session_id_hash"].unique())
        log(f"train: {train.shape}")
        log(f"test: {test.shape}")
        log(f"sku_to_content: {sku_to_content.shape}")

    with span("Preprocess data"):
        pr = Preprocessor(config)
        train_preprocessed, test_preprocessed= pr.run(train, test, sku_to_content)
        log(f"train_preprocessed: {train_preprocessed.shape}")
        log(f"test_preprocessed: {test_preprocessed.shape}")

    with span("Get CV"):
        train_session_info = train_preprocessed[
            ["session_id_hash", "session_len_count"]
        ].drop_duplicates()
        cv = StratifiedKFold(**config["fold_params"])
        folds = cv.split(
            train_session_info,
            pd.cut(
                train_session_info["session_len_count"],
                config["fold_params"]["n_splits"],
                labels=False,
            ),
        )

    log("Training")
    for i_fold, (trn_idx, val_idx) in enumerate(folds):
        if holdout and i_fold > 0:
            break

        with span(f"Fold = {i_fold}"):
            train_session_ids = train_session_info.iloc[trn_idx]["session_id_hash"].tolist()
            val_session_ids = train_session_info.iloc[val_idx]["session_id_hash"].tolist()
            train_session_seqs = pr.get_session_sequences(
                train_preprocessed[train_preprocessed["session_id_hash"].isin(train_session_ids)]
            )
            val_session_seqs = pr.get_session_sequences(
                train_preprocessed[train_preprocessed["session_id_hash"].isin(val_session_ids)]
            )
            log(f"number of train sessions: {len(train_session_seqs)}")
            log(f"number of valid sessions: {len(val_session_seqs)}")

            dataset = RecTaskDataModule(config, train_session_seqs, val_session_seqs)
            dataset.train_dataloader()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--holdout", action="store_true")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
        config["exp_name"] = Path(f.name).stem

    output_path = Path(config["file_path"]["output_dir"]) / config["exp_name"]
    if not output_path.exists():
        output_path.mkdir(parents=True)
    set_out(output_path / "train_log.txt")

    seed_everything(config["seed"])
    log(f"configuration: {config}")
    run(config, args.debug, args.holdout)
