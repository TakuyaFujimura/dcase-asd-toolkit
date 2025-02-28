# DCASE Anomalous Sound Detection Library

This repository provides various recipes for the DCASE Task2 Anomalous Sound Detection (ASD).



## Easy Start

<details>
<summary>1. Clone and install this repository</summary>
<br>

**How to**

```bash
[somewhere]$ git clone https://github.com/TakuyaFujimura/dcase-asd-library.git
[somewhere]$ cd dcase-asd-library
[dcase-asd-library]$ python3 -m venv venv # Python 3.10+ required
[dcase-asd-library]$ source venv/bin/activate
[dcase-asd-library]$ pip install -e .
```

</details>

<details>
<summary>2. Download DCASETask2 dataset</summary>
<br>

**How to**
- Specify `data_dir` and `dcase` in `jobs/download/run.sh`
- `data_dir`: The directory where the dataset is stored
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
- This will create a formatted dataset by making symbolic links to the original dataset (while keeping the original dataset)
- The ground truth normal/anomalous labels are added during this process

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
<summary>4. Execute training/testing recipe</summary>

<br>

**How to**
- Specify `data_dir` in `dcase-asd-library/config/config.yaml`
- Specify `dcase` in `jobs/asd/example/dis_baseline.sh`
- This will automatically execute training/testing process
- An example script is provided but you can also create your own configuration file (see [Customization](docs/customization.md))

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
                    └─── valve
                        ├── hparams.yaml
                        ├── test_evaluate.csv # AUCs on test data
                        ├── test_extraction.csv # information of test data including embedding values
                        ├── test_score.csv # anomaly scores of test data
                        ├── train_extraction.csv # information of train data including embedding values
                        ├── train_score.csv # anomaly scores of train data
                        ├── umap.csv # UMAP embedding values
                        └── umap_*.png # UMAP visualization
                
```

</details>


## Information

### References

### Author

Takuya Fujimura, Toda Labotorary, Nagoya University  
E-mail: fujimura.takuya@g.sp.m.is.nagoya-u.ac.jp
