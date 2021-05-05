import torch
import torch.nn as nn
import pytorch_lightning as pl
from src.models.transformer_encoder import TransformerEncoderModel


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
        return loss

    def validation_step(self, batch, batch_idx):
        x_batch, y_batch = batch
        y_pred = self.forward(x_batch)
        metric = 0
        return metric

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            **self.config["optimizer_params"],
        )
        return optimizer
