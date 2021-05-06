import pytorch_lightning as pl
from pathlib import Path


def get_trainer(config: dict) -> pl.Trainer:
    ckpt_callback = pl.callbacks.ModelCheckpoint(
        dirpath=Path(config["file_path"]["output_dir"]) / config["exp_name"],
        filename='{epoch:03d}-{val_loss:.3f}-{val_mrr:.3f}-{val_f1_score:.3f}',
        save_top_k=1,
        monitor="val_mrr",
        mode="max",
    )
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor="val_mrr",
        min_delta=0.00,
        patience=3,
        verbose=True,
        mode="max",
    )
    return pl.Trainer(
        max_epochs=config["n_epochs"],
        callbacks=[ckpt_callback, early_stop_callback],
        gpus=1,
    )
