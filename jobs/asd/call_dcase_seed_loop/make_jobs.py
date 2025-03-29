from pathlib import Path

head = """#!/bin/bash
#PJM -L rscgrp=cx-single
#PJM -L gpu=4
#PJM -L elapse=12:00:00
#PJM -j
# ---------------------------- #
"""

bottom = """recipe="${recipe_stem}.sh"
cd ../../recipe

mkdir ../logs/${recipe_stem}

for dcase in "${dcase_list[@]}"; do
    CUDA_VISIBLE_DEVICES=0 bash "${recipe}" "${dcase}" "0" > "../logs/${recipe_stem}/${dcase}_0.log" 2>&1 &
    CUDA_VISIBLE_DEVICES=1 bash "${recipe}" "${dcase}" "1" > "../logs/${recipe_stem}/${dcase}_1.log" 2>&1 &
    CUDA_VISIBLE_DEVICES=2 bash "${recipe}" "${dcase}" "2" > "../logs/${recipe_stem}/${dcase}_2.log" 2>&1 &
    CUDA_VISIBLE_DEVICES=3 bash "${recipe}" "${dcase}" "3" > "../logs/${recipe_stem}/${dcase}_3.log" 2>&1 &
    wait
done
"""


def main(method: str):
    save_dir = Path(f"./{method}")
    save_dir.mkdir(parents=True, exist_ok=False)
    for dcase in ["dcase2021", "dcase2022", "dcase2023", "dcase2024"]:
        dcase_txt = f'dcase_list=("{dcase}")'
        dcase_txt += "\n# ---------------------------- #"
        recipe_txt = f'recipe_stem="{method}"'

        sh_txt = "\n".join([head, dcase_txt, recipe_txt, bottom])
        with open(save_dir / f"{dcase}.sh", "w") as f:
            f.write(sh_txt)


if __name__ == "__main__":
    for method in ["adacos", "scac_trainable", "scac_trainable_ms", "sl"]
        main(f"dis_baseline_{method}")
    main("dis_multires_scac_trainable")
