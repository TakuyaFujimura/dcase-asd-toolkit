# DCASE Anomalous Sound Detection Library

This repository provides various recipes for the DCASE Task 2 Anomalous Sound Detection (ASD).

## Easy Start

<details>
<summary>1. Clone and install this repository</summary>
<br>

**How to**

```bash
[somewhere]$ git clone https://github.com/TakuyaFujimura/dcase-asd-library.git
[somewhere]$ cd dcase-asd-library
[dcase-asd-library]$ python3 -m venv venv # Requires Python 3.10+
[dcase-asd-library]$ source venv/bin/activate
[dcase-asd-library]$ pip install -e .
```

</details>

<details>
<summary>2. Download the DCASE Task 2 dataset</summary>
<br>

**How to**
- Specify `data_dir` and `dcase` in `jobs/download/run.sh`
- `data_dir`: The directory where the dataset will be stored. The default is set to the parent directory of this repository. If unchanged, you do not need to modify `data_dir` in other scripts.
- `dcase`: The dataset name (`dcase2021`, `dcase2022`, `dcase2023`, `dcase2024` are available)

```bash
[dcase-asd-library]$ cd jobs/download
[dcase-asd-library/jobs/download]$ bash run.sh
```

**Result**

```bash
<data_dir>
└── original
    ├── <dcase>
    └── ...
```
</details>

<details>
<summary>3. Format the dataset</summary>
<br>

**How to**

- Specify `data_dir` and `dcase` in `jobs/format/run.sh`
- This process creates a formatted dataset by generating symbolic links to the original dataset (without modifying the original files).
- Normal/anomalous ground truth labels for test data are added during this process.

```bash
[dcase-asd-library]$ cd jobs/format
[dcase-asd-library/jobs/format]$ bash run.sh
```

**Result**

```bash
<data_dir>
├── original
│   ├── <dcase>
│   └── ...
└── formatted
    ├── <dcase>
    └── ...
```

</details>

<details>
<summary>4. Execute the training/testing recipe</summary>
<br>

**How to**
- Specify `data_dir` in `config/train/config.yaml`
- Specify `dcase` in `jobs/asd/example/?.sh`
- This process will automatically execute the training and testing pipeline.
- Two example scripts are provided: `dis_baseline.sh` and `ae_baseline.sh`


```bash
[dcase-asd-library]$ cd jobs/asd/example
[dcase-asd-library/jobs/asd/example]$ bash dis_baseline.sh
```

**Result**
```bash
dcase-asd-library
├── asdlib
├── ...
└── results
    ├── ...
    └── <name> # `example`
        ├── ...
        └── <version> # `dcase2023_baseline_0`
            ├── model
            │   └── <model_ver> # all
            │       ├── .hydra
            │       ├── checkpoints
            │       ├── events.out.tfevents.*
            │       ├── hparams.yaml
            │       └── train.log
            └── output
                └── <ckpt_ver> # `epoch_12`
                    ├── bandsaw
                    ├── bearing
                    ├── ...
                    └── valve
                        ├── hparams.yaml
                        ├── test_evaluate.csv # AUC scores on test data
                        ├── test_extraction.csv # Extracted test data information, including embedding values
                        ├── test_score.csv # Anomaly scores for test data
                        ├── train_extraction.csv # Extracted training data information, including embedding values
                        ├── train_score.csv # Anomaly scores for training data
                        ├── umap.csv # UMAP embedding values
                        └── umap_*.png # UMAP visualization
```

</details>

## Information

### References

### Author

Takuya Fujimura, Toda Laboratory, Nagoya University  
E-mail: fujimura.takuya@g.sp.m.is.nagoya-u.ac.jp
