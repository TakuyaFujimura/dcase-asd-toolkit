from pathlib import Path

head = """#!/bin/bash
#PJM -L rscgrp=cx-single
#PJM -L gpu=4
#PJM -L elapse=12:00:00
#PJM -j
# ---------------------------- #
"""

bottom = """cd ../../recipe

mkdir ../logs/${recipe}

for dcase in "${dcase_list[@]}"; do
    CUDA_VISIBLE_DEVICES=0 bash dis_any_version.sh "${dcase}" "0" "${recipe}" > "../logs/${recipe}/${dcase}_0.log" 2>&1 &
    CUDA_VISIBLE_DEVICES=1 bash dis_any_version.sh "${dcase}" "1" "${recipe}" > "../logs/${recipe}/${dcase}_1.log" 2>&1 &
    CUDA_VISIBLE_DEVICES=2 bash dis_any_version.sh "${dcase}" "2" "${recipe}" > "../logs/${recipe}/${dcase}_2.log" 2>&1 &
    CUDA_VISIBLE_DEVICES=3 bash dis_any_version.sh "${dcase}" "3" "${recipe}" > "../logs/${recipe}/${dcase}_3.log" 2>&1 &
    wait
done
"""


def main(method: str):
    save_dir = Path(f"./{method}")
    save_dir.mkdir(parents=True, exist_ok=False)
    for dcase in ["dcase2021", "dcase2022", "dcase2023", "dcase2024"]:
        dcase_txt = f'dcase_list=("{dcase}")'
        dcase_txt += "\n# ---------------------------- #"
        recipe_txt = f'recipe="{method}"'

        sh_txt = "\n".join([head, dcase_txt, recipe_txt, bottom])
        with open(save_dir / f"{dcase}.sh", "w") as f:
            f.write(sh_txt)


if __name__ == "__main__":
    for method in ["ms", "adacos", "scac_trainable", "scac_trainable_ms"]:
        main(f"dis_baseline_{method}")
    # main("dis_multires_scac_trainable")
    # for i in {1..4}; do pjsub "dcase202${i}.sh"; done
