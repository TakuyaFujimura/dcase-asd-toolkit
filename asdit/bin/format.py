import argparse
import logging
from pathlib import Path

import tqdm

from asdit.utils.asdit_utils.format import (
    RenameTestPath,
    check_src_dir,
    get_traintest_dir_dict,
)
from asdit.utils.dcase_utils import MACHINE_DICT

logger = logging.getLogger(__name__)


def main(dcase: str, data_dir: str, link_mode: str) -> None:
    src_dir = Path(data_dir) / "original" / dcase
    check_src_dir(src_dir=src_dir, dcase=dcase)
    dst_dir = Path(data_dir) / "formatted" / dcase

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
                        dst_wav_path.parent.mkdir(parents=True, exist_ok=True)
                        if link_mode == "symlink":
                            dst_wav_path.symlink_to(src_wav_path)
                        elif link_mode == "mv":
                            src_wav_path.rename(dst_wav_path)
                        else:
                            raise ValueError(f"Unknown mkdir_mode: {link_mode}.")


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
