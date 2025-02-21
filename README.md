### Preperation

Please refer each README in the corresponding directories for the detailed instructions.

1. Download Dataset (preprocess/download)
2. Formatting Dataset (preprocess/formatting)
3. Create Virtual Environment
    - `[dcase_task2]$ python -m venv venv`
    - `[dcase_task2]$ source venv/bin/activate`
    - `(venv) [dcase_task2]$ pip install -r requirements.txt`
4. Execute Training/Testing (jobs)


<details>
<summary>Dataset structure</summary>

```bash
<data_dir>
├── original
│   ├── dcase2021
│   ├── dcase2022
│   └── ...
└── formatted
    ├── dcase2021
    ├── dcase2022
    └── ...
```
<!-- ```bash
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
``` -->
</details>
