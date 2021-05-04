import pandas as pd
from pathlib import Path
from typing import Tuple


class Preprocessor:
    def __init__(self, config: dict) -> None:
        self.config = config
        self.interim_dir = Path(config["file_path"]["interim_dir"])
        self.encode_cols = [
            "session_id_hash",
            "product_sku_hash",
        ]
        self.label_to_index_dict = {}
        self.index_to_label_dict = {}

    def run(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        sku_to_content: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        train["is_test"] = False
        test["is_test"] = True
        total = pd.concat([train, test], axis=0)

        self._label_encoding(total)
        total = self._filter_out(total)

        train_preprocessed = total[total["is_test"] == False]
        test_preprocessed = total[total["is_test"] == True]
        train_preprocessed.to_pickle(self.interim_dir / self.config["pkl_file"]["train_preprocessed"])
        test_preprocessed.to_pickle(self.interim_dir / self.config["pkl_file"]["test_preprocessed"])
        return train_preprocessed, test_preprocessed

    def _label_encoding(self, df: pd.DataFrame) -> None:
        for col in self.encode_cols:
            index_series, label_to_index, index_to_label = self._label_encode_series(df[col])
            df[col] = index_series
            self.label_to_index_dict[col] = label_to_index
            self.index_to_label_dict[col] = index_to_label

    @staticmethod
    def _label_encode_series(series: pd.Series) -> Tuple[pd.Series, dict, dict]:
        """https://github.com/coveooss/SIGIR-ecom-data-challenge/blob/main/baselines/create_session_rec_input.py#L31-L42
        """
        labels = set(series.unique())
        label_to_index = {l: idx for idx, l in enumerate(labels) if l == l}   # avoid null value
        index_to_label = {v: k for k, v in label_to_index.items()}
        return series.map(label_to_index), label_to_index, index_to_label

    @staticmethod
    def _filter_out(df: pd.DataFrame) -> pd.DataFrame:
        # `remove from cart` events to avoid feeding them to session_rec as positive signals
        df = df[df['product_action'] != 'remove']
        # rows with null product_sku_hash
        df = df.dropna(subset=['product_sku_hash'])
        return df
