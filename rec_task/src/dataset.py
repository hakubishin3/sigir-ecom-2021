import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict, List, NamedTuple


class Example(NamedTuple):
    """https://github.com/sakami0000/kaggle_riiid/blob/main/src/data.py
    """
    input_ids: torch.LongTensor
    attention_mask: torch.LongTensor
    server_timestamp_epoch_sec: torch.FloatTensor

    def to(self, device: torch.device) -> "Example":
        return Example(
            input_ids=self.input_ids.to(device),
            attention_mask=self.attention_mask.to(device),
            server_timestamp_epoch_sec=self.server_timestamp_epoch_sec.to(device),
        )

    def to_dict(self) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids,
            attention_mask=self.attention_mask,
            server_timestamp_epoch_sec=self.server_timestamp_epoch_sec,
        )


class RecTaskDataset(Dataset):
    """https://github.com/sakami0000/kaggle_riiid/blob/main/src/data.py
    """
    def __init__(
        self,
        session_seqs: Dict[int, Dict[str, List[int]]],
        window_size: int = 10,
    ) -> None:
        self.session_seqs = session_seqs
        self.all_examples = []
        self.all_targets = []

        for session_seq in self.session_seqs.values():
            sequence_length = len(session_seq["product_sku_hash"])
            end_idx = sequence_length - 1
            start_idx = max(0, end_idx - window_size)

            product_sku_hash = session_seq["product_sku_hash"][start_idx:end_idx]
            server_timestamp_epoch_sec = session_seq["server_timestamp_epoch_sec"][start_idx:end_idx]
            target = session_seq["product_sku_hash"][-1]

            if len(product_sku_hash) < window_size:
                # padding
                pad_size = window_size - len(product_sku_hash)
                product_sku_hash += [0] * pad_size
                server_timestamp_epoch_sec += [0] * pad_size

            input_ids = torch.LongTensor(product_sku_hash)
            attention_mask = (input_ids > 0).float()
            server_timestamp_epoch_sec = torch.FloatTensor(server_timestamp_epoch_sec)

            example = Example(
                input_ids=input_ids,
                attention_mask=attention_mask,
                server_timestamp_epoch_sec=server_timestamp_epoch_sec,
            )

            self.all_examples.append(example)
            self.all_targets.append(torch.LongTensor([target]))

    def __len__(self):
        return len(self.session_seqs)

    def __getitem__(self, idx: int) -> Tuple[Example, torch.Tensor]:
        return self.all_examples[idx], self.all_targets[idx]


class RecTaskDataModule(pl.LightningDataModule):
    def __init__(
        self,
        config: dict,
        train_session_seqs: Dict[int, Dict[str, List[int]]],
        val_session_seqs: Dict[int, Dict[str, List[int]]],
    ) -> None:
        super().__init__()
        self.config = config
        self.train_session_seqs = train_session_seqs
        self.val_session_seqs = val_session_seqs

    def train_dataloader(self) -> "DataLoader":
        train_dataset = RecTaskDataset(
            session_seqs=self.train_session_seqs,
            window_size=self.config["window_size"],
        )
        train_loader = DataLoader(
            train_dataset,
            **self.config["train_loader_params"],
        )
        return train_loader

    def val_dataloader(self) -> "DataLoader":
        val_dataset = RecTaskDataset(
            session_seqs=self.val_session_seqs,
            window_size=self.config["window_size"],
        )
        val_loader = DataLoader(
            val_dataset,
            **self.config["val_loader_params"],
        )
        return val_loader