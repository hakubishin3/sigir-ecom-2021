import yaml
import argparse
from pathlib import Path

from src import log, set_out, span
from src.utils import seed_everything
from src.data_loader import DataLoader
from src.preprocessor import Preprocessor


def run(config: dict, debug: bool) -> None:
    with span("Load datasets"):
        train, test, sku_to_content = DataLoader(config, debug).load_datasets()
        test_session_ids = set(test["session_id_hash"].unique())
        log(f"train: {train.shape}")
        log(f"test: {test.shape}")
        log(f"sku_to_content: {sku_to_content.shape}")

    with span("Preprocess data"):
        train_preprocessed, test_preprocessed = Preprocessor(config).run(train, test, sku_to_content)
        log(f"train_preprocessed: {train_preprocessed.shape}")
        log(f"test_preprocessed: {test_preprocessed.shape}")
    import pdb; pdb.set_trace()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
        config["exp_name"] = Path(f.name).stem

    output_path = Path(config["file_path"]["output_dir"]) / config["exp_name"]
    if not output_path.exists():
        output_path.mkdir(parents=True)
    set_out(output_path / "train_log.txt")

    seed_everything(config["seed"])
    log(f"configuration: {config}")
    run(config, args.debug)
