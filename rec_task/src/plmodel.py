import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import List

from src.models.transformer_encoder import TransformerEncoderModel
from src.metrics import evaluate_rec_task_metrics


class RecTaskPLModel(pl.LightningModule):
    def __init__(self, config: dict, num_labels: int) -> None:
        super().__init__()
        self.config = config
        self.model = TransformerEncoderModel(
            encoder_params=config["encoder_params"],
            num_labels=num_labels,
        )
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x_batch):
        output = self.model(**x_batch.to(self.config["device"]).to_dict())
        return output

    def training_step(self, batch, batch_idx):
        x_batch, y_batch = batch
        y_pred = self.forward(x_batch)
        loss = self.criterion(y_pred, y_batch.squeeze_())

        self.log("loss", loss, prog_bar=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x_batch, y_batch = batch
        y_pred = self.forward(x_batch)
        metrics = evaluate_rec_task_metrics(y_pred, y_batch)
        loss = self.criterion(y_pred, y_batch.squeeze_())
        metrics["loss"] = loss

        self.log("val_loss", loss)
        self.log_dict(metrics)
        return metrics

    def training_epoch_end(self, train_step_outputs: List[dict]):
        loss = torch.stack([o["loss"] for o in train_step_outputs]).mean()

        self.log("loss", loss)

    def validation_epoch_end(self, val_step_outputs: List[dict]):
        val_loss = torch.stack([o["loss"] for o in val_step_outputs]).mean()
        val_f1_score = torch.stack([o["f1_score"] for o in val_step_outputs]).mean()
        val_mrr = torch.stack([o["mrr"] for o in val_step_outputs]).mean()

        self.log("val_loss", val_loss)
        self.log("val_f1_score", val_f1_score)
        self.log("val_mrr", val_mrr)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            **self.config["optimizer_params"],
        )
        return optimizer
