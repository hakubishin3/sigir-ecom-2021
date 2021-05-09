import torch
import numpy as np
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict, List, NamedTuple


def get_onehot(tensor: torch.Tensor, num_labels: int) -> torch.Tensor:
    tensor_onehot = nn.functional.one_hot(
        tensor, num_labels
    ).max(dim=0)[0].float()
    return tensor_onehot


class Example(NamedTuple):
    """https://github.com/sakami0000/kaggle_riiid/blob/main/src/data.py
    """
    input_ids: torch.LongTensor
    attention_mask: torch.LongTensor
    elapsed_time: torch.FloatTensor
    product_action: torch.LongTensor
    hashed_url: torch.LongTensor
    price_bucket: torch.LongTensor
    number_of_category_hash: torch.LongTensor
    category_hash_first_level: torch.LongTensor
    category_hash_second_level: torch.LongTensor
    category_hash_third_level: torch.LongTensor
    description_vector: torch.FloatTensor
    image_vector: torch.FloatTensor

    def to_dict(self, device: torch.device) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids.to(device),
            attention_mask=self.attention_mask.to(device),
            elapsed_time=self.elapsed_time.to(device),
            product_action=self.product_action.to(device),
            hashed_url=self.hashed_url.to(device),
            price_bucket=self.price_bucket.to(device),
            number_of_category_hash=self.number_of_category_hash.to(device),
            category_hash_first_level=self.category_hash_first_level.to(device),
            category_hash_second_level=self.category_hash_second_level.to(device),
            category_hash_third_level=self.category_hash_third_level.to(device),
            description_vector=self.description_vector.to(device),
            image_vector=self.image_vector.to(device),
        )


class RecTaskDataset(Dataset):
    """https://github.com/sakami0000/kaggle_riiid/blob/main/src/data.py
    """
    def __init__(
        self,
        session_seqs: Dict[int, Dict[str, List[int]]],
        num_labels: int,
        window_size: int,
        is_test: bool = False,
        max_output_size: int = 20,
    ) -> None:
        self.session_seqs = session_seqs
        self.num_labels = num_labels
        self.is_test = is_test
        self.max_output_size = max_output_size
        self.all_examples = []
        self.all_targets = []

        for session_seq in self.session_seqs.values():
            sequence_length = len(session_seq["product_sku_hash"])

            if not self.is_test:
                max_output_size = min(self.max_output_size, sequence_length - 1)
                n_output = np.random.randint(1, max_output_size + 1)
                end_idx = sequence_length - n_output
            else:
                n_output = 1
                end_idx = sequence_length
            start_idx = max(0, end_idx - window_size)

            product_sku_hash = session_seq["product_sku_hash"][start_idx:end_idx]
            target = session_seq["product_sku_hash"][-1 * n_output:]

            pad_size = window_size - len(product_sku_hash)
            product_sku_hash += [0] * pad_size
            elapsed_time = session_seq["elapsed_time"][start_idx:end_idx] + [0] * pad_size
            product_action = session_seq["product_action"][start_idx:end_idx] + [0] * pad_size
            hashed_url = session_seq["hashed_url"][start_idx:end_idx] + [0] * pad_size
            price_bucket = session_seq["price_bucket"][start_idx:end_idx] + [0] * pad_size
            number_of_category_hash = session_seq["number_of_category_hash"][start_idx:end_idx] + [0] * pad_size
            category_hash_first_level = session_seq["category_hash_first_level"][start_idx:end_idx] + [0] * pad_size
            category_hash_second_level = session_seq["category_hash_second_level"][start_idx:end_idx] + [0] * pad_size
            category_hash_third_level = session_seq["category_hash_third_level"][start_idx:end_idx] + [0] * pad_size
            description_vector = session_seq["description_vector"][start_idx:end_idx] + [[0] * 50] * pad_size
            image_vector = session_seq["image_vector"][start_idx:end_idx] + [[0] * 50] * pad_size

            input_ids = torch.LongTensor(product_sku_hash)
            attention_mask = (input_ids > 0).float()
            target = torch.LongTensor(target)

            example = Example(
                input_ids=input_ids,
                attention_mask=attention_mask,
                elapsed_time=torch.LongTensor(elapsed_time),
                product_action=torch.LongTensor(product_action),
                hashed_url=torch.LongTensor(hashed_url),
                price_bucket=torch.LongTensor(price_bucket),
                number_of_category_hash=torch.LongTensor(number_of_category_hash),
                category_hash_first_level=torch.LongTensor(category_hash_first_level),
                category_hash_second_level=torch.LongTensor(category_hash_second_level),
                category_hash_third_level=torch.LongTensor(category_hash_third_level),
                description_vector=torch.FloatTensor(description_vector),
                image_vector=torch.FloatTensor(image_vector),
            )

            self.all_examples.append(example)
            self.all_targets.append(target)

    def __len__(self):
        return len(self.session_seqs)

    def __getitem__(self, idx: int) -> Tuple[Example, torch.Tensor]:
        if not self.is_test:
            target_onehot_subsequent_items = get_onehot(self.all_targets[idx], self.num_labels)
            target_onehot_next_item = get_onehot(self.all_targets[idx][:1], self.num_labels)
            return self.all_examples[idx], target_onehot_next_item, target_onehot_subsequent_items
        else:
            return self.all_examples[idx]


class RecTaskDataModule(pl.LightningDataModule):
    def __init__(
        self,
        config: dict,
        train_session_seqs: Dict[int, Dict[str, List[int]]],
        val_session_seqs: Dict[int, Dict[str, List[int]]],
        test_session_seqs: Dict[int, Dict[str, List[int]]],
        num_labels: int,
    ) -> None:
        super().__init__()
        self.config = config
        self.train_session_seqs = train_session_seqs
        self.val_session_seqs = val_session_seqs
        self.test_session_seqs = test_session_seqs
        self.num_labels = num_labels

    def train_dataloader(self) -> "DataLoader":
        train_dataset = RecTaskDataset(
            session_seqs=self.train_session_seqs,
            num_labels=self.num_labels,
            window_size=self.config["window_size"],
            is_test=False,
        )
        train_loader = DataLoader(
            train_dataset,
            **self.config["train_loader_params"],
        )
        return train_loader

    def val_dataloader(self) -> "DataLoader":
        val_dataset = RecTaskDataset(
            session_seqs=self.val_session_seqs,
            num_labels=self.num_labels,
            window_size=self.config["window_size"],
            is_test=False,
        )
        val_loader = DataLoader(
            val_dataset,
            **self.config["val_loader_params"],
        )
        return val_loader

    def test_dataloader(self) -> "DataLoader":
        test_dataset = RecTaskDataset(
            session_seqs=self.test_session_seqs,
            num_labels=self.num_labels,
            window_size=self.config["window_size"],
            is_test=True,
        )
        test_loader = DataLoader(
            test_dataset,
            **self.config["test_loader_params"],
        )
        return test_loader
