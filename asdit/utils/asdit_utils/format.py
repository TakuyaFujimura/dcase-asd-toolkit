from collections import defaultdict
from itertools import chain
from pathlib import Path
from typing import Dict

from asdit.utils.dcase_utils import MACHINE_DICT


def get_traintest_dir_dict(dcase: str) -> Dict[str, list]:
    """Return expected subdirectories"""
    if dcase in ["dcase2020", "dcase2022", "dcase2023", "dcase2024"]:
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
    """Check if src_dir has the correct structure."""
    if not src_dir.exists():
        raise FileNotFoundError(f"{src_dir} does not exist.")

    traintest_dir_dict = get_traintest_dir_dict(dcase=dcase)

    for split_de in ["dev", "eval"]:
        split_de_dir = src_dir / f"{split_de}_data/raw"
        est_machines = [d.name for d in split_de_dir.iterdir() if d.is_dir()]
        ref_machines = MACHINE_DICT[f"{dcase}-{split_de}"]  # type: ignore
        if sorted(est_machines) != sorted(ref_machines):
            raise ValueError(
                f"{split_de_dir} does not contain correct machines ({ref_machines})."
            )

        for machine in ref_machines:
            for split_tt in chain(*traintest_dir_dict.values()):
                split_tt_dir = split_de_dir / machine / split_tt
                if not split_tt_dir.exists():
                    raise FileNotFoundError(f"{split_tt_dir} does not exist.")


class RenameTestPath:
    """Rename wav_path to a unified format and append ground-truth normal/anomaly label."""

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

    def postprocess(self, wav_path: Path) -> Path:
        """This is postprocess function for dcase2020."""
        if self.dcase != "dcase2020":
            return wav_path

        # dcase2020 wav_path #################################################
        # original
        # dcase2020/dev_data/raw/fan/train/normal_id_00_00000001.wav
        # formatted
        # dcase2020/raw/fan/train/section_00_source_train_normal_00000001.wav
        ######################################################################
        split_tt = wav_path.parents[0].name
        split_name = wav_path.name.split("_")
        new_name = "_".join(
            [
                "section",
                split_name[2],
                "source",
                split_tt,
                split_name[0],
                split_name[-1],
            ]
        )
        return wav_path.parent / new_name

    def __call__(self, wav_path: Path) -> Path:
        """
        Change the name of wav_path. Parents path is not changed.
        """
        if wav_path.parents[4].name != self.dcase:
            raise ValueError(f"{wav_path} is not in {self.dcase}.")

        split_tt = wav_path.parents[0].name
        split_de = wav_path.parents[3].name

        # check train/test
        if split_tt == "train":
            return self.postprocess(wav_path)
        elif split_tt not in ["test", "source_test", "target_test"]:
            raise ValueError(f"Unknown split_tt: {split_tt}.")

        # check dev/eval
        if split_de == "dev_data":
            return self.postprocess(wav_path)
        elif split_de != "eval_data":
            raise ValueError(f"Unknown split_de: {split_de}.")

        # path is in eval_data/test
        machine = wav_path.parents[1].name
        renamed_wav_path = wav_path.parent / self.path_dict[machine][wav_path.name]
        return self.postprocess(renamed_wav_path)
