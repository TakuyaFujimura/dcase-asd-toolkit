import argparse
import logging
from pathlib import Path

import tqdm

from asdkit.utils.asdkit_utils.format import (
    RenameTestPath,
    arrange_audio_file,
    check_src_dir,
    get_split_dir_names_dict,
    iter_formatted_paths,
)
from asdkit.utils.dcase_utils import MACHINE_DICT

logger = logging.getLogger(__name__)


def main(dcase: str, data_dir: str, link_mode: str) -> None:
    data_root = Path(data_dir).absolute()
    src_dir = data_root / "original" / dcase
    check_src_dir(src_dir=src_dir, dcase=dcase)
    dst_dir = data_root / "formatted" / dcase
    if dst_dir.exists():
        raise FileExistsError(f"{dst_dir} already exists.")

    split_dir_names_dict = get_split_dir_names_dict(dcase=dcase)
    renamer = RenameTestPath(dcase=dcase, metadata_dir=data_root / "metadata")
    for subset in ["dev", "eval"]:
        for machine in tqdm.tqdm(MACHINE_DICT[f"{dcase}-{subset}"]):  # type: ignore
            for src_wav_path, dst_wav_path in iter_formatted_paths(
                src_machine_dir=src_dir / f"{subset}_data/raw/{machine}",
                dst_machine_dir=dst_dir / "raw" / machine,
                split_dir_names_dict=split_dir_names_dict,
                renamer=renamer,
            ):
                arrange_audio_file(
                    src_wav_path=src_wav_path,
                    dst_wav_path=dst_wav_path,
                    link_mode=link_mode,
                )


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--data_dir", type=str)
    args.add_argument("--dcase", type=str)
    args.add_argument(
        "--link_mode",
        type=str,
        choices=["symlink", "mv"],
        default="symlink",
        help="How to arrange files: use symbolic links or move the actual files.",
    )
    args = args.parse_args()
    main(dcase=args.dcase, data_dir=args.data_dir, link_mode=args.link_mode)
