# Preperation
## Dataset
1. Download the dataset from the DCASE website and extract the files to the arbitrary `<data_dir>` directory.
2. Execute `preprocess/dataset/dcase??.sh` to reorganize the dataset. This script will create a new directory in `<data_dir>` but will not modify the original dataset.

<details>
<summary>Dataset structure</summary>

```bash
<data_dir>
├── dcase2020 # TODO: check the directory structure
├── dcase2021 # TODO: check the directory structure
├── dcase2022 # TODO: check the directory structure
├── dcase2023
│   ├── all # Created by `preprocess/dataset/dcase2023.sh`
│   │   └── raw
│   │       ├── ToyCar
│   │       │   ├── train
│   │       │   └── test
│   │       └── ToyCircuit
│   ├── dev_data  # Original structure
│   └── eval_data # Original structure
│       
│
└── dcase2024 # TODO: check the directory structure
```
</details>


1. preprocess/download
2. preprocess/formatting
3. preprocess/labeling
4. jobs/example/exec0.sh
