import argparse
from pathlib import Path

from asdlib.utils.io_utils.json_util import read_json

MACHINE_DICT = read_json("preprocess/format/dataset/machines.json")


def check_src_dir(src_dir: Path, dcase: str):
    if not src_dir.exists():
        raise FileNotFoundError(f"{src_dir} does not exist.")

    for split in ["dev", "eval"]:
        split_dir = src_dir / f"{split}_data/raw"
        est_machines = [d.name for d in split_dir.iterdir() if d.is_dir()]
        ref_machines = MACHINE_DICT[f"{dcase}-{split}"]
        if sorted(est_machines) != sorted(ref_machines):
            raise ValueError(f"{split_dir} does not contain correct machines.")

        for machine in ref_machines:
            train_dir = split_dir / machine / "train"
            if not train_dir.exists():
                raise FileNotFoundError(f"{train_dir} does not exist.")

            if dcase in ["dcase2022", "dcase2023", "dcase2024"]:
                test_dir = split_dir / machine / "test"
                if not test_dir.exists():
                    raise FileNotFoundError(f"{test_dir} does not exist.")
            elif dcase in ["dcase2021"]:
                source_test_dir = split_dir / machine / "source_test"
                target_test_dir = split_dir / machine / "target_test"
                if not source_test_dir.exists():
                    raise FileNotFoundError(f"{source_test_dir} does not exist.")
                if not target_test_dir.exists():
                    raise FileNotFoundError(f"{target_test_dir} does not exist.")
            else:
                raise ValueError(f"Unknown dcase: {dcase}.")


def cp_train(src_dir: Path, dst_dir: Path, dcase: str):
    for split in ["dev", "eval"]:
        split_dir = src_dir / f"{split}_data/raw"
        for machine in MACHINE_DICT[f"{dcase}-{split}"]:
            train_dir = split_dir / machine / "train"
            for src_wav_path in train_dir.glob("*.wav"):
                dst_wav_path = dst_dir / f"raw/{machine}/{src_wav_path.name}"
                dst_wav_path.parent.mkdir(parents=True, exist_ok=True)
                dst_wav_path.symlink_to(src_wav_path)


def main(data_dir: str, dcase: str):
    src_dir = Path(data_dir) / "original" / dcase
    check_src_dir(src_dir=src_dir, dcase=dcase)
    dst_dir = Path(data_dir) / "formatted" / dcase
    # dst_dir.mkdir(parents=True, exist_ok=True)
    # cp_train(src_dir=src_dir, dst_dir=dst_dir, dcase=dcase)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--data_dir", type=str)
    args.add_argument("--dcase", type=str)
    args = args.parse_args()
    main(data_dir=args.data_dir, dcase=args.dcase)
