import torch
import numpy as np
import pandas as pd
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
    elapsed_time: torch.LongTensor
    event_type: torch.LongTensor
    product_action: torch.LongTensor
    hashed_url: torch.LongTensor
    price_bucket: torch.LongTensor
    number_of_category_hash: torch.LongTensor
    category_hash_first_level: torch.LongTensor
    category_hash_second_level: torch.LongTensor
    category_hash_third_level: torch.LongTensor
    description_vector: torch.FloatTensor
    image_vector: torch.FloatTensor
    hour: torch.LongTensor
    weekday: torch.LongTensor
    weekend: torch.LongTensor
    query_vector: torch.FloatTensor

    def to_dict(self, device: torch.device) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids.to(device),
            attention_mask=self.attention_mask.to(device),
            elapsed_time=self.elapsed_time.to(device),
            event_type=self.event_type.to(device),
            product_action=self.product_action.to(device),
            hashed_url=self.hashed_url.to(device),
            price_bucket=self.price_bucket.to(device),
            number_of_category_hash=self.number_of_category_hash.to(device),
            category_hash_first_level=self.category_hash_first_level.to(device),
            category_hash_second_level=self.category_hash_second_level.to(device),
            category_hash_third_level=self.category_hash_third_level.to(device),
            description_vector=self.description_vector.to(device),
            image_vector=self.image_vector.to(device),
            hour=self.hour.to(device),
            weekday=self.weekday.to(device),
            weekend=self.weekend.to(device),
            query_vector=self.query_vector.to(device),
        )


class RecTaskDataset(Dataset):
    """https://github.com/sakami0000/kaggle_riiid/blob/main/src/data.py
    """
    def __init__(
        self,
        session_seqs: Dict[int, Dict[str, List[int]]],
        output_type: str,
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
        self.output_type = output_type
        seqs = []
        outs = []
        items = []

        if output_type == "subsequent_items":
            self.thres_n_items = 2
        elif output_type == "next_item":
            self.thres_n_items = 1
        else:
            raise NameError

        for session_seq in self.session_seqs.values():
            # session_seq["product_sku_hash"]
            # ex. [view, detail, view, detail add, view]
            sequence_length = len(session_seq["product_sku_hash"])

            if not self.is_test:
                # train and validate mode
                if sequence_length == 1:
                    # cant make input or target
                    continue

                n_items = len(
                    np.unique([i for i in session_seq["product_sku_hash"] if i != 1])
                )   # 1 is nan

                if n_items < self.thres_n_items:
                    # cant make target
                    continue

                if n_items == 1 and session_seq["product_sku_hash"][0] != 1:
                    # ex. [detail, view, view]
                    # cant make input
                    continue

                idx_dict = {}
                for end_idx in range(1, len(session_seq["product_sku_hash"])):
                    if session_seq["product_sku_hash"][end_idx] == 1:
                        continue
                    start_idx = max(0, end_idx - window_size)
                    input_items = set(session_seq["product_sku_hash"][start_idx:end_idx]) - {1}
                    output_items = set(session_seq["product_sku_hash"][end_idx:]) - {1} - input_items
                    if len(output_items) < self.thres_n_items:
                        continue
                    idx_dict[end_idx] = len(output_items)

                if len(idx_dict) == 0:
                    continue

                n_output = np.random.choice(list(set(idx_dict.values())))
                candidacies_end_idx = []
                for k, v in idx_dict.items():
                    if v == n_output:
                        candidacies_end_idx.append(k)

                end_idx = max(candidacies_end_idx)
            else:
                # test mode
                end_idx = sequence_length

            start_idx = max(0, end_idx - window_size)

            product_sku_hash = session_seq["product_sku_hash"][start_idx:end_idx]

            pad_size = window_size - len(product_sku_hash)
            product_sku_hash = [0] * pad_size + product_sku_hash
            elapsed_time = [0] * pad_size + session_seq["elapsed_time"][start_idx:end_idx]
            event_type = [0] * pad_size + session_seq["event_type"][start_idx:end_idx] 
            product_action = [0] * pad_size + session_seq["product_action"][start_idx:end_idx] 
            hashed_url = [0] * pad_size + session_seq["hashed_url"][start_idx:end_idx] 
            price_bucket = [0] * pad_size + session_seq["price_bucket"][start_idx:end_idx] 
            number_of_category_hash = [0] * pad_size + session_seq["number_of_category_hash"][start_idx:end_idx] 
            category_hash_first_level = [0] * pad_size + session_seq["category_hash_first_level"][start_idx:end_idx] 
            category_hash_second_level = [0] * pad_size + session_seq["category_hash_second_level"][start_idx:end_idx] 
            category_hash_third_level = [0] * pad_size + session_seq["category_hash_third_level"][start_idx:end_idx] 
            description_vector = [[0] * 50] * pad_size + session_seq["description_vector"][start_idx:end_idx]
            image_vector = [[0] * 50] * pad_size + session_seq["image_vector"][start_idx:end_idx]
            hour = [0] * pad_size + session_seq["hour"][start_idx:end_idx]
            weekday = [0] * pad_size + session_seq["weekday"][start_idx:end_idx]
            weekend = [0] * pad_size + session_seq["weekend"][start_idx:end_idx]
            query_vector = [[0] * 50] * pad_size + session_seq["query_vector"][start_idx:end_idx]

            if not self.is_test:
                target = [i for i in session_seq["product_sku_hash"][end_idx:] if i != 1 and i not in product_sku_hash]   # remove nan
                target = sorted(set(target), key=target.index)
                if self.output_type == "subsequent_items":
                    target = target[:20]
                elif self.output_type == "next_item":
                    target = target[:1]
            else:
                target = [0]   # tekitou

            input_ids = torch.LongTensor(product_sku_hash)
            attention_mask = (input_ids > 0).float()
            target = torch.LongTensor(target)

            example = Example(
                input_ids=input_ids,
                attention_mask=attention_mask,
                elapsed_time=torch.LongTensor(elapsed_time),
                event_type=torch.LongTensor(event_type),
                product_action=torch.LongTensor(product_action),
                hashed_url=torch.LongTensor(hashed_url),
                price_bucket=torch.LongTensor(price_bucket),
                number_of_category_hash=torch.LongTensor(number_of_category_hash),
                category_hash_first_level=torch.LongTensor(category_hash_first_level),
                category_hash_second_level=torch.LongTensor(category_hash_second_level),
                category_hash_third_level=torch.LongTensor(category_hash_third_level),
                description_vector=torch.FloatTensor(description_vector),
                image_vector=torch.FloatTensor(image_vector),
                hour=torch.LongTensor(hour),
                weekday=torch.LongTensor(weekday),
                weekend=torch.LongTensor(weekend),
                query_vector=torch.FloatTensor(query_vector),
            )

            self.all_examples.append(example)
            self.all_targets.append(target)

            seqs.append(len(session_seq["product_sku_hash"][start_idx:end_idx]))
            outs.append(target.size()[0])
            items.append(len(set(session_seq["product_sku_hash"][start_idx:end_idx]) - {1}))
        seqs = pd.Series(seqs).value_counts().sort_index()
        outs = pd.Series(outs).value_counts().sort_index()
        items = pd.Series(items).value_counts().sort_index()
        print("seqs")
        print(seqs.sum())
        print(seqs / seqs.sum())
        print("outs")
        print(outs.sum())
        print(outs / outs.sum())
        print("items")
        print(items.sum())
        print(items / items.sum())

    def __len__(self):
        return len(self.all_examples)

    def __getitem__(self, idx: int) -> Tuple[Example, torch.Tensor]:
        if not self.is_test:
            target = get_onehot(self.all_targets[idx], self.num_labels)
            return self.all_examples[idx], target
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
        self.output_type = config["output_type"]

    def train_dataloader(self) -> "DataLoader":
        train_dataset = RecTaskDataset(
            session_seqs=self.train_session_seqs,
            output_type=self.output_type,
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
            output_type=self.output_type,
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
            output_type=self.output_type,
            num_labels=self.num_labels,
            window_size=self.config["window_size"],
            is_test=True,
        )
        test_loader = DataLoader(
            test_dataset,
            **self.config["test_loader_params"],
        )
        return test_loader
