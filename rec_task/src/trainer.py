import pytorch_lightning as pl


def get_trainer(config: dict) -> pl.Trainer:
    return pl.Trainer(
        max_epochs=config["n_epochs"],
        profiler="simple",
        gpus=1,
    )
