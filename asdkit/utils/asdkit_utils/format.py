from collections import defaultdict
from itertools import chain
from pathlib import Path
from shlex import quote
from typing import Dict, Iterator, List, Tuple

from asdkit.utils.asdkit_utils.eval_metadata import get_eval_data_list_path
from asdkit.utils.dcase_utils import MACHINE_DICT


def get_split_dir_names_dict(dcase: str) -> Dict[str, List[str]]:
    """Map formatted splits to their original directory names."""
    if dcase in ["dcase2025"]:
        split_dir_names_dict = {
            "train": ["train"],
            "test": ["test"],
            "supplemental": ["supplemental"],
        }
    elif dcase in ["dcase2020", "dcase2022", "dcase2023", "dcase2024", "dcase2026"]:
        split_dir_names_dict = {"train": ["train"], "test": ["test"]}
    elif dcase in ["dcase2021"]:
        split_dir_names_dict = {
            "train": ["train"],
            "test": ["source_test", "target_test"],
        }
    else:
        raise ValueError(f"Unknown dcase: {dcase}.")
    return split_dir_names_dict


def check_src_dir(src_dir: Path, dcase: str) -> None:
    """Check if src_dir has the correct structure."""
    if not src_dir.exists():
        raise FileNotFoundError(f"{src_dir} does not exist.")

    split_dir_names_dict = get_split_dir_names_dict(dcase=dcase)

    for subset in ["dev", "eval"]:
        subset_dir = src_dir / f"{subset}_data/raw"
        est_machines = [
            d.name
            for d in subset_dir.iterdir()
            if d.is_dir() and d.name != ".completed"
        ]
        ref_machines = MACHINE_DICT[f"{dcase}-{subset}"]  # type: ignore
        if sorted(est_machines) != sorted(ref_machines):
            raise ValueError(
                f"{subset_dir} does not contain correct machines ({ref_machines})."
            )

        for machine in ref_machines:
            for source_split_dir_name in chain(*split_dir_names_dict.values()):
                source_split_dir = subset_dir / machine / source_split_dir_name
                if not source_split_dir.exists():
                    raise FileNotFoundError(f"{source_split_dir} does not exist.")


class RenameTestPath:
    """Rename wav_path to a unified format and append ground-truth normal/anomaly label."""

    def __init__(self, dcase: str, metadata_dir: Path) -> None:
        # data/original/dcase2024/dev_data/raw/bearing/test/hoge.wav
        self.dcase = dcase
        self.path_dict = defaultdict(dict)  # type: ignore
        csv_path = get_eval_data_list_path(dcase=dcase, metadata_dir=metadata_dir)
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Evaluation metadata not found: {csv_path}\n"
                "Download it first from the repository root:\n"
                "python -m asdkit.bin.download_eval_data "
                f"dcase={dcase} {quote(f'data_dir={metadata_dir.parent}')}"
            )
        with csv_path.open(encoding="utf-8") as f:
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

        # DCASE 2020 filename conversion:
        # normal_id_00_00000001.wav -> section_00_source_train_normal_00000001.wav
        source_split_dir_name = wav_path.parents[0].name
        filename_parts = wav_path.name.split("_")
        new_name = "_".join(
            [
                "section",
                filename_parts[2],
                "source",
                source_split_dir_name,
                filename_parts[0],
                filename_parts[-1],
            ]
        )
        return wav_path.parent / new_name

    def __call__(self, wav_path: Path) -> Path:
        """
        Change the name of wav_path. Parents path is not changed.
        """
        if wav_path.parents[4].name != self.dcase:
            raise ValueError(f"{wav_path} is not in {self.dcase}.")

        source_split_dir_name = wav_path.parents[0].name
        subset_dir_name = wav_path.parents[3].name

        # Check train/test
        if source_split_dir_name in ["train", "supplemental"]:
            return self.postprocess(wav_path)
        elif source_split_dir_name not in ["test", "source_test", "target_test"]:
            raise ValueError(f"Unknown source_split_dir_name: {source_split_dir_name}.")

        # Check dev/eval
        if subset_dir_name == "dev_data":
            return self.postprocess(wav_path)
        elif subset_dir_name != "eval_data":
            raise ValueError(f"Unknown subset_dir_name: {subset_dir_name}.")

        # Look up evaluation test labels
        machine = wav_path.parents[1].name
        renamed_wav_path = wav_path.parent / self.path_dict[machine][wav_path.name]
        return self.postprocess(renamed_wav_path)


def iter_formatted_paths(
    src_machine_dir: Path,
    dst_machine_dir: Path,
    split_dir_names_dict: Dict[str, List[str]],
    renamer: RenameTestPath,
) -> Iterator[Tuple[Path, Path]]:
    """Yield source and destination paths in the original traversal order."""
    for destination_split, source_split_dir_names in split_dir_names_dict.items():
        for source_split_dir_name in source_split_dir_names:
            for source_path in (src_machine_dir / source_split_dir_name).glob("*.wav"):
                destination_name = renamer(wav_path=source_path).name
                yield source_path, dst_machine_dir / destination_split / destination_name


def arrange_audio_file(src_wav_path: Path, dst_wav_path: Path, link_mode: str) -> None:
    """Create the destination directory and link or move one audio file."""
    dst_wav_path.parent.mkdir(parents=True, exist_ok=True)
    if link_mode == "symlink":
        dst_wav_path.symlink_to(src_wav_path)
    elif link_mode == "mv":
        src_wav_path.rename(dst_wav_path)
    else:
        raise ValueError(f"Unknown link_mode: {link_mode}.")
