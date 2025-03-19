import argparse
from collections import defaultdict
from itertools import chain
from pathlib import Path
from typing import Dict

import tqdm

from asdit.utils.dcase_utils import MACHINE_DICT


def mk_symlink(link_path: Path, target_path: Path) -> None:
    link_path.parent.mkdir(parents=True, exist_ok=True)
    link_path.symlink_to(target_path)


def get_traintest_dir_dict(dcase: str) -> Dict[str, list]:
    if dcase in ["dcase2022", "dcase2023", "dcase2024"]:
        traintest_dir_dict = {"train": ["train"], "test": ["test"]}
    elif dcase in ["dcase2021"]:
        traintest_dir_dict = {
            "train": ["train"],
            "test": ["source_test", "target_test"],
        }
    else:
        raise ValueError(f"Unknown dcase: {dcase}.")
    return traintest_dir_dict


def check_src_dir(src_dir: Path, dcase: str):
    if not src_dir.exists():
        raise FileNotFoundError(f"{src_dir} does not exist.")

    traintest_dir_dict = get_traintest_dir_dict(dcase=dcase)

    for split_de in ["dev", "eval"]:
        split_de_dir = src_dir / f"{split_de}_data/raw"
        est_machines = [d.name for d in split_de_dir.iterdir() if d.is_dir()]
        ref_machines = MACHINE_DICT[f"{dcase}-{split_de}"]  # type: ignore
        if sorted(est_machines) != sorted(ref_machines):
            raise ValueError(f"{split_de_dir} does not contain correct machines.")

        for machine in ref_machines:
            for split_tt in chain(*traintest_dir_dict.values()):
                split_tt_dir = split_de_dir / machine / split_tt
                if not split_tt_dir.exists():
                    raise FileNotFoundError(f"{split_tt_dir} does not exist.")


class RenameTestPath:
    def __init__(self, dcase: str):
        # data/original/dcase2024/dev_data/raw/bearing/test/hoge.wav
        self.dcase = dcase
        self.path_dict = defaultdict(dict)  # type: ignore
        with open(f"data/eval_data_list_{dcase[5:]}.csv", "r") as f:
            for line in f:
                csv_data_list = line.strip().split(",")
                if not csv_data_list[0].endswith(".wav"):
                    current_machine = csv_data_list[0]
                    continue
                else:
                    assert csv_data_list[0].endswith(".wav")
                    assert csv_data_list[1].endswith(".wav")
                    self.path_dict[current_machine][csv_data_list[0]] = csv_data_list[1]

    def __call__(self, wav_path: Path) -> Path:
        if wav_path.parents[4].name != self.dcase:
            raise ValueError(f"{wav_path} is not in {self.dcase}.")

        split_tt = wav_path.parents[0].name
        split_de = wav_path.parents[3].name

        # check train/test
        if split_tt == "train":
            return wav_path
        elif split_tt not in ["test", "source_test", "target_test"]:
            raise ValueError(f"Unknown split_tt: {split_tt}.")

        # check dev/eval
        if split_de == "dev_data":
            return wav_path
        elif split_de != "eval_data":
            raise ValueError(f"Unknown split_de: {split_de}.")

        # path is in eval_data/test
        machine = wav_path.parents[1].name
        renamed_wav_path = wav_path.parent / self.path_dict[machine][wav_path.name]
        return renamed_wav_path


def cp_dataset(src_dir: Path, dst_dir: Path, dcase: str):
    traintest_dir_dict = get_traintest_dir_dict(dcase=dcase)
    renamer = RenameTestPath(dcase=dcase)
    for split_de in ["dev", "eval"]:
        for machine in tqdm.tqdm(MACHINE_DICT[f"{dcase}-{split_de}"]):  # type: ignore
            for split_tt_dst, split_tt_src_list in traintest_dir_dict.items():
                for split_tt_src in split_tt_src_list:
                    src_wav_dir = (
                        src_dir / f"{split_de}_data/raw/{machine}/{split_tt_src}"
                    )

                    for src_wav_path in src_wav_dir.glob("*.wav"):
                        renamed_src_wav_name = renamer(wav_path=src_wav_path).name
                        dst_wav_path = (
                            dst_dir
                            / f"raw/{machine}/{split_tt_dst}/{renamed_src_wav_name}"
                        )
                        mk_symlink(link_path=dst_wav_path, target_path=src_wav_path)


def main():
    args = argparse.ArgumentParser()
    args.add_argument("--data_dir", type=str)
    args.add_argument("--dcase", type=str)
    args = args.parse_args()
    data_dir = args.data_dir
    dcase = args.dcase
    src_dir = Path(data_dir) / "original" / dcase
    check_src_dir(src_dir=src_dir, dcase=dcase)
    dst_dir = Path(data_dir) / "formatted" / dcase
    cp_dataset(src_dir=src_dir, dst_dir=dst_dir, dcase=dcase)


if __name__ == "__main__":
    main()
