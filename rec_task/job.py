import yaml
import argparse
from pathlib import Path

from src import log, set_out, span
from src.utils import seed_everything


def run(config: dict, debug: bool) -> None:
    pass


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
    run(config, args.debug)
