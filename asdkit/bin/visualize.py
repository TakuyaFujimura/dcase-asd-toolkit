# Copyright 2024 Takuya Fujimura

import logging
import warnings

import hydra
import lightning.pytorch as pl
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from asdkit.utils.asdkit_utils.path import get_output_dir
from asdkit.utils.asdkit_utils.visualize import load_npz_dict, plot_umap, trans_umap
from asdkit.utils.common.hydra import hydra_to_pydantic
from asdkit.utils.config_class import MainVisualizeConfig

logger = logging.getLogger(__name__)

warnings.filterwarnings(
    "ignore",
    message=".*n_jobs value 1 overridden to 1 by setting random_state. Use no seed for parallelism..*",
)
warnings.filterwarnings(
    "ignore",
    message=".*'force_all_finite' was renamed to 'ensure_all_finite' in 1.6 and will be removed in 1.8..*",
)


@hydra.main(version_base=None, config_path="../../config/visualize", config_name="main")
def main(hydra_cfg: DictConfig) -> None:
    cfg: MainVisualizeConfig = hydra_to_pydantic(
        hydra_cfg=hydra_cfg, cls=MainVisualizeConfig
    )
    logger.info(f"Start visualization: {HydraConfig().get().run.dir}")
    pl.seed_everything(seed=0, workers=True)

    output_dir = get_output_dir(cfg)
    visualize_dir = output_dir / "visualize"
    visualize_dir.mkdir(parents=True, exist_ok=True)

    path_stem = f"umap_{cfg.metric}_{cfg.embed_key}"

    # UMAP transformation
    if not (visualize_dir / f"{path_stem}.npz").exists() or cfg.overwrite:
        logger.info(f"Start UMAP transformation: {visualize_dir}")
        trans_umap(
            output_dir=output_dir,
            metric=cfg.metric,
            embed_key=cfg.embed_key,
            save_path=visualize_dir / f"{path_stem}.npz",
            extract_items=cfg.extract_items,
        )
    else:
        logger.info(
            f"Read existing {visualize_dir}/{path_stem}.npz. "
            + "Set overwrite=True to overwrite it."
        )

    # Visualization
    visualize_dict = load_npz_dict(visualize_dir / f"{path_stem}.npz")
    logger.info(f"Start UMAP visualization: {visualize_dir}")
    plot_umap(
        visualize_dict=visualize_dict, save_dir=visualize_dir, save_path_stem=path_stem
    )
    logger.info(f"Finished UMAP visualization: {visualize_dir}")


if __name__ == "__main__":
    main()
