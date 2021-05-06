import yaml
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold

from src import log, set_out, span
from src.utils import seed_everything
from src.data_loader import DataLoader
from src.preprocessor import Preprocessor
from src.dataset import RecTaskDataModule
from src.plmodel import RecTaskPLModel
from src.trainer import get_trainer
from src.submission import submission


def run(config: dict, debug: bool, holdout: bool) -> None:
    seed_everything(config["seed"])

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
    num_labels = len(pr.index_to_label_dict["product_sku_hash"]) + 1   # plus padding id
    test_session_seqs = pr.get_session_sequences(test_preprocessed)
    test_pred_all_folds = np.zeros((len(test_session_seqs), num_labels), dtype=np.float32)

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

            dataset = RecTaskDataModule(
                config,
                train_session_seqs,
                val_session_seqs,
                test_session_seqs,
            )
            model = RecTaskPLModel(config, num_labels=num_labels)
            trainer = get_trainer(config)
            trainer.fit(model, dataset)
            best_ckpt = (
                Path(config["file_path"]["output_dir"])
                / config["exp_name"]
                / f"best_model_fold{i_fold}.ckpt"
            )
            trainer.save_checkpoint(best_ckpt)

            _test_pred = trainer.predict(model, dataset.test_dataloader())   # batch_size = 1
            test_pred = np.array([i.reshape(-1) for i in _test_pred])
            test_pred_all_folds += test_pred / config["fold_params"]["n_splits"]

    with span("Make submission file"):
        session_id_hash_index_to_test_data_index = {}
        for i, session_id_hash_index in enumerate(test_session_seqs.keys()):
            session_id_hash_index_to_test_data_index[session_id_hash_index] = i

        raw_file_path = Path(config["file_path"]["input_dir"]) / config["raw_file"]["test"]
        with raw_file_path.open() as f:
            original_test_data = json.load(f)

        popular_item_index = (
            train_preprocessed.groupby("product_sku_hash")
            .size()
            .sort_values(ascending=False)
            .head(20)
            .index
        )
        popular_item_list = []
        for item_index in popular_item_index:
            product_sku_hash = pr.index_to_label_dict["product_sku_hash"][item_index]
            popular_item_list.append(product_sku_hash)

        for idx, query_label in enumerate(original_test_data):
            query = query_label["query"]
            session_id_hash = query[0]["session_id_hash"]
            if pr.label_to_index_dict["session_id_hash"].get(session_id_hash) is None:
                original_test_data[idx]["label"] = popular_item_list
            else:
                session_id_hash_index = pr.label_to_index_dict["session_id_hash"][session_id_hash]
                test_data_index = session_id_hash_index_to_test_data_index[session_id_hash_index]
                pred = test_pred[test_data_index]
                items_index = pred.argsort()[-20:]
                item_list = []
                for item_index in items_index:
                    product_sku_hash = pr.index_to_label_dict["product_sku_hash"][item_index]
                    item_list.append(product_sku_hash)
                original_test_data[idx]["label"] = item_list

        outfile_path = Path(config["file_path"]["output_dir"]) / config["exp_name"] / "submission.json"
        with outfile_path.open("w") as outfile:
            json.dump(original_test_data, outfile)

    with span("Submit"):
        submission(outfile_path, config["task"])


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

    log(f"configuration: {config}")
    run(config, args.debug, args.holdout)
