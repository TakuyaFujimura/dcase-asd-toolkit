from pathlib import Path

import hydra
from omegaconf import DictConfig
from pydantic import BaseModel

from asdkit.utils.asdkit_utils.eval_metadata import prepare_eval_data_list
from asdkit.utils.common.hydra import hydra_to_pydantic


class Config(BaseModel):
    data_dir: Path
    dcase: str


@hydra.main(
    version_base=None, config_path="../../config/download_eval_data", config_name="main"
)
def main(hydra_cfg: DictConfig) -> None:
    cfg = hydra_to_pydantic(hydra_cfg=hydra_cfg, cls=Config)
    csv_path = prepare_eval_data_list(
        dcase=cfg.dcase, metadata_dir=cfg.data_dir.absolute() / "metadata"
    )
    print(f"Evaluation metadata ready: {csv_path}")


if __name__ == "__main__":
    main()
