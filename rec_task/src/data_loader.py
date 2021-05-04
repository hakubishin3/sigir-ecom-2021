import ast
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple


class DataLoader:
    def __init__(self, config: dict) -> None:
        self.config = config
        self.input_dir = Path(config["file_path"]["input_dir"])

    def load_datasets(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        browsing_train = self.load_train_data("browsing_train")
        search_train = self.load_train_data("search_train")
        sku_to_content = self.load_train_data("sku_to_content")
        return (
            browsing_train,
            search_train,
            sku_to_content,
        )

    def load_train_data(self, data_type: str) -> pd.DataFrame:
        raw_file_path = self.input_dir / self.config["raw_file"][data_type]
        pkl_file_path = self.input_dir / self.config["pkl_file"][data_type]
        if pkl_file_path.exists():
            dataset = pd.read_pickle(pkl_file_path)
        else:
            dataset = pd.read_csv(raw_file_path)
            if data_type == "browsing_train":
                self._preprocessed_browsing_train(dataset)
            elif data_type == "search_train":
                self._preprocessed_search_train(dataset)
            elif data_type == "sku_to_content":
                self._preprocessed_sku_to_content(dataset)
            dataset.to_pickle(pkl_file_path)
        return dataset

    @staticmethod
    def _preprocessed_browsing_train(df: pd.DataFrame) -> None:
        df["server_timestamp"] = pd.to_datetime(
            df["server_timestamp_epoch_ms"], unit="ms",
        )

    @staticmethod
    def _preprocessed_search_train(df: pd.DataFrame) -> None:
        df["server_timestamp"] = pd.to_datetime(
            df["server_timestamp_epoch_ms"], unit="ms",
        )
        df["query_vector"] = (
            df["query_vector"]
            .apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else np.nan)
        )
        df["product_skus_hash"] = (
            df["product_skus_hash"]
            .apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else np.nan)
        )
        df["clicked_skus_hash"] = (
            df["clicked_skus_hash"]
            .apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else np.nan)
        )

    @staticmethod
    def _preprocessed_sku_to_content(df: pd.DataFrame) -> None:
        df["image_vector"] = (
            df["image_vector"]
            .apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else np.nan)
        )
        df["description_vector"] = (
            df["description_vector"]
            .apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else np.nan)
        )
        df["category_hash_first_level"] = (
            df["category_hash"]
            .apply(lambda x: x.split("/")[0] if isinstance(x, str) else np.nan)
        )
        df["category_hash_second_level"] = (
            df["category_hash"]
            .apply(lambda x: x.split("/")[1] if isinstance(x, str) and len(x.split("/")) >= 2 else np.nan)
        )
        df["category_hash_third_level"] = (
            df["category_hash"]
            .apply(lambda x: x.split("/")[2] if isinstance(x, str) and len(x.split("/")) >= 3 else np.nan)
        )
