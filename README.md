# 🧰 ASDKit: A Toolkit for Comprehensive Evaluation of Anomalous Sound Detection Methods ⛏️

[![arXiv](https://img.shields.io/badge/arXiv-2507.10264-b31b1b.svg)](https://arxiv.org/abs/2507.10264)

ASDKit is a toolkit for anomalous sound detection (ASD) task.
<br>
ASDKit provides recipes for various ASD methods and supports evaluation on the DCASE 2020-2026 Task 2 datasets.


## 📢 Update History

#### October 2026: Major Update to v1.0.0

ASDKit v1.0.0 updates the framework and adds new methods. See the [release notes](./docs/v1.0.0.md) for details.

Models trained with v0.0.0 can be used after [checkpoint conversion](./asdkit/legacy/v0_to_v1/README.md) although some inference behavior has changed; see [Inference Differences](./asdkit/legacy/v0_to_v1/README.md#inference-differences).

The previous version remains available in the [v0.0.0 release](https://github.com/TakuyaFujimura/dcase-asd-toolkit/releases/tag/v0.0.0).

<details>
<summary>New Features</summary>

- **Backends:** [Relative Deviation Pooling (RDP)](https://arxiv.org/abs/2603.04605), [BEAM, AdaBEAM](https://arxiv.org/abs/2603.13749), and [variance-minimization-based rescaling](https://dcase.community/documents/workshop2025/proceedings/DCASE2025Workshop_Matsumoto_12.pdf)
- **SSL Models:**
    - [Dasheng](https://arxiv.org/abs/2406.06992)
    - [EAT](https://huggingface.co/worstchan/EAT-base_epoch30_pretrain) without fairseq
    - See [Optional dependencies and pre-trained model usage](#optional-dependencies-and-pre-trained-model-usage) for setup instructions.


</details>

<details>
<summary>Previous Updates</summary>

#### July 21, 2025: Added pseudo labeling

- [Details](./special/pseudo_label/README.md)
- https://arxiv.org/abs/2607.16678

</details>

## 🚀 Easy Start

<details>
<summary>1. Clone and install this repository</summary>
<br>

**How to**

```bash
[somewhere]$ git clone https://github.com/TakuyaFujimura/dcase-asd-toolkit.git
[somewhere]$ cd dcase-asd-toolkit
```

Choose one of the following installation methods.

**Option 1. Use uv**

If you want to use another CUDA version or CPU builds, please change the PyTorch index of `pyproject.toml` first.

```bash
[dcase-asd-toolkit] uv lock # Generate uv.lock based on pyproject.toml
[dcase-asd-toolkit] uv sync --locked # Sync the environment with uv.lock
```

The `uv.lock` file will be provided later.

**Option 2. Other installation methods**

You can also install ASDKit using `uv pip`, `pip`, or other installation methods.
ASDKit's job scripts activate the repository's `.venv` environment via `jobs/activate.sh`.
If you create an environment elsewhere or use a different environment manager, modify `jobs/activate.sh` to activate your environment.


</details>

<details>
<summary><a id="optional-dependencies-and-pre-trained-model-usage"></a>(Optional dependencies and pre-trained model usage)</summary>
<br>

**How to**

Install extras for the features you use. For example, for the EAT model, use the `eat` extra:

```bash
uv sync --locked --extra eat
```
or
```bash
pip install -e '.[eat]'
```
or other commands.

To install all optional dependencies:

```bash
uv sync --locked --all-extras
```

**List of optional dependencies**

| Model | Setup |
| --- | --- |
| EAT | Use the `eat` extra. |
| Dasheng | Use the `dasheng` extra. |
| BEATs | Download a [BEATs checkpoint](https://github.com/microsoft/unilm/tree/master/beats) (e.g., `BEATs_iter3.pt`) and place it in `pretrained_models/beats/`. No extra installation is needed. |


| Other Feature | Setup |
| --- | --- |
| UMAP visualization | Use the `visualize` extra. |


</details>


<details>
<summary>2. Download the DCASE Task 2 datasets</summary>
<br>

**How to**
- Specify both `data_dir` and `dcase` in `jobs/download/run.sh`.
- `data_dir`: The directory where the dataset will be stored. By default, it is set to the parent directory of this repository. If you change this, make sure to update `data_dir` in the other scripts accordingly.
- `dcase`: The name of the dataset. Available options are: `dcase2020`, `dcase2021`, `dcase2022`, `dcase2023`, `dcase2024`, `dcase2025`, and `dcase2026`.

```bash
# You can run the following script from anywhere
[somewhere]$ bash /path/to/dcase-asd-toolkit/jobs/download/run.sh
```

**Result**

```bash
<data_dir>
└── original
    └── <dcase>
```
</details>

<details>
<summary>3. Format the dataset</summary>
<br>

This process creates a formatted dataset in which the filename format and dataset structure are unified across all DCASE versions.  
Additionally, it assigns ground-truth normal/anomalous labels to the test data. These labels are concealed during the challenge and released by the organizers afterward.


**How to**

- Specify `data_dir`, `dcase`, and `link_mode` in `jobs/format/run.sh`.
- `link_mode`:  
    - If set `link_mode` to `symlink`, symbolic links with formatted filenames will be created. This mode does not modify the original files.  
    - If set to `link_mode` to `mv`, the original files will be directly renamed using the formatted filenames.
- On Windows, creating symbolic links needs to be allowed (```Settings > System > For Developers```).


```bash
# You can run the following script from anywhere
[somewhere]$ bash /path/to/dcase-asd-toolkit/jobs/format/run.sh
```

**Result**

```bash
<data_dir>
├── original
│   └── <dcase>
└── formatted
    └── <dcase>
```


</details>


<details>
<summary>4. Execute the training and evaluation recipe</summary>
<br>

This script automatically performs the training and evaluation processes.

**How to**
- Specify `data_dir` in `config/train/main.yaml` and `config/extract/main.yaml`.
- Specify `dcase` in `jobs/asd/run/?.sh` (and `seed` if needed).
- Scripts in `jobs/asd/run` serve as wrappers for recipes in `jobs/asd/recipe`.
    - Recipe scripts activate the virtual environment using `jobs/activate.sh`.
    - To visualize embeddings using UMAP, install the `visualize` extra and add `asdkit_visualize` after `asdkit_extract` in the recipe.


```bash
# You can run the following script from anywhere
[somewhere]$ bash /path/to/dcase-asd-toolkit/jobs/asd/run/?.sh
```

**Result**
```bash
dcase-asd-toolkit
├── asdkit
├── ...
└── results
    └── <name> # `recipe`
        └── <dcase>
            └── <version> # `dis_spec_scac_trainable`
                └── <seed> # `0`
                    ├── model
                    │   └── <model_ver> # all
                    │       ├── .hydra
                    │       ├── checkpoints/
                    │       ├── events.out.tfevents.*
                    │       ├── hparams.yaml
                    │       └── train.log
                    └── output
                        └── <infer_ver> # `default`
                            ├── bandsaw
                            ├── bearing
                            ├── ...
                            ├── valve
                            │   ├── test_evaluate.csv # Evaluation results
                            │   ├── test_extract/ # Extracted test data information
                            │   ├── test_extract.metadata.json # Frontend output metadata
                            │   ├── test_score.csv # Anomaly scores for test data
                            │   ├── train_extract/ # Extracted training data information
                            │   ├── train_extract.metadata.json # Frontend output metadata
                            │   ├── train_score.csv # Anomaly scores for training data
                            │   └── visualize/ # Only when optional visualization is enabled
                            ├── dev_official23.csv # Summarized evaluation results for dev subsets
                            ├── ...
                            └── eval_official23.csv # Summarized evaluation results for eval subsets
```

</details>


## 📋 Supported Experimental Conditions
ASDKit supports the DCASE 2020-2026 conditions.
Across these conditions, the machine types, the provided meta-information labels, and the evaluation scores differ.
For all conditions, the datasets are divided into official *dev* and *eval* subsets, and evaluation scores are aggregated separately for each subset.
The evaluation score is based on a combination of several types of AUC.
ASDKit automatically computes the official evaluation scores according to the respective DCASE year.

The figure below illustrates the DCASE 2020-2024 conditions.

<img src="docs/conditions.png" width="1000">

### Supported Evaluation Metrics

| Metrics       | Description |
|---------------|-------------|
| s_auc / s_pauc | AUC / pAUC computed using normal and anomalous sounds in the source domain. |
| t_auc / t_pauc | AUC / pAUC computed using normal and anomalous sounds in the target domain. |
| mix_auc / mix_pauc | AUC / pAUC computed using normal and anomalous sounds from both the source and target domains. |
| smix_auc       | AUC computed using normal and anomalous sounds in the source domain and anomalous sounds in the target domain. |
| tmix_auc       | AUC computed using normal and anomalous sounds in the target domain and anomalous sounds in the source domain. |
| official20-26  | Official evaluation score for DCASE 2020-2026 Challenge Task2. These are computed combining the above metrics. |



## ⚒️ Supported Methods
ASDKit handles various ASD methods in a unified framework.
All ASD methods consist of *frontend* and *backend* modules, where the frontend extracts some features from audio signals, and the backend computes anomaly scores after processing these features.
<br>
Recipes in `jobs/asd/recipe/` define the processing flow within this framework.
<br>
Run scripts in `jobs/asd/run/` serve as wrappers for these recipes, allowing easy execution with different seeds, datasets, and configurations.


<img src="docs/framework.png" width="500">

<br>
In the following, we explain the ASD methods supported by ASDKit and their processing pipelines within this four-step framework.

<br>

<details>
<summary>Autoencoder</summary>
<br>

- [AE-based official DCASE baseline](https://dcase.community/documents/workshop2020/proceedings/DCASE2020Workshop_Koizumi_3.pdf)
- Description: This method trains an AE network to reconstruct audio features of normal sounds. It computes anomaly scores based on the reconstruction error for a given observation.
- Run Script: `jobs/asd/run/dcase_ae.sh`
- Recipe: `jobs/asd/recipe/gen_any.sh`
    -  This method directly computes anomaly scores based on the reconstruction error of encountered test samples in the second step, and in the third step, the backend simply copies these anomaly scores.
    - The AE is independently trained for each machine type, repeating steps 1-4 for each machine type.
```bash
# jobs/asd/recipe/gen_any.sh (version=dcase_ae)

experiments_train="dcase_ae" # config/train/experiments/dcase_ae.yaml
experiments_extract="from_checkpoint/machinewise" # config/extract/experiments/from_checkpoint/machinewise.yaml
experiments_score="no_backend" # config/score/experiments/no_backend.yaml

for machine in "${machines[@]}"; do
    asdkit_train machine="${machine}" experiments="${experiments_train}"
    asdkit_extract experiments="${experiments_extract}"
    asdkit_score experiments="${experiments_score}"
    asdkit_evaluate
done
```

</details>



<details>
<summary>Discriminative Methods</summary>
<br>

|  Component | Supported Options |
|:------|:-----|
| Frontend | [Dual-branch CNN](https://ieeexplore.ieee.org/document/10097176), [Multi-branch CNN](https://ieeexplore.ieee.org/document/10890020), [BEATs and EAT w/ LoRA](https://ieeexplore.ieee.org/document/10832335) |
| Loss Function | [ArcFace](https://ieeexplore.ieee.org/document/8953658), [AdaCos](https://ieeexplore.ieee.org/document/8953896), [Sub-cluster AdaCos](https://ieeexplore.ieee.org/document/9534290), [AdaProj](https://dcase.community/documents/workshop2024/proceedings/DCASE2024Workshop_Wilkinghoff_1.pdf) |
| Data Augmentation | [Mixup](https://arxiv.org/abs/1710.09412), [SpecAug](https://www.isca-archive.org/interspeech_2019/park19e_interspeech.pdf) |
| Other Techniques | [FeatEx](https://ieeexplore.ieee.org/document/10447156), [Subspace Loss](https://ieeexplore.ieee.org/document/10890020) |
| Backend | kNN, [kNN w/ Kmeans](https://ieeexplore.ieee.org/document/10097176), [kNN w/ SMOTE](https://ieeexplore.ieee.org/document/10889514), [kNN w/ Rescaling](https://ieeexplore.ieee.org/document/10888402), [variance-minimization-based rescaling](https://dcase.community/documents/workshop2025/proceedings/DCASE2025Workshop_Matsumoto_12.pdf), [BEAM and AdaBEAM](https://arxiv.org/abs/2603.13749) |


- Description: This approach trains a discriminative feature extractor to classify meta-information labels associated with normal training data, and then computes anomaly scores in the discriminative feature space.
- Run Script: `jobs/asd/run/dis_*.sh`
- Recipe: `jobs/asd/recipe/dis_any.sh`
    - The discriminative methods train a common frontend using data from all machine types, and then independently train the backend for each machine type by repeating steps 2-4 for each machine type.

```bash
# jobs/asd/recipe/dis_any.sh

experiments_train="dis_*" # config/train/experiments/dis_*.yaml
experiments_extract="from_checkpoint/shared" # config/extract/experiments/from_checkpoint/shared.yaml
experiments_score="default" # config/score/experiments/default.yaml

asdkit_train experiments="${experiments_train}"
for machine in "${machines[@]}"; do
    asdkit_extract experiments="${experiments_extract}"
    asdkit_score experiments="${experiments_score}"
    asdkit_evaluate
done
```
</details>


<details>
<summary>Raw Feature Methods</summary>
<br>


|  Component | Supported Options |
|:------|:-----|
| Frontend | [Spectrogram](https://ieeexplore.ieee.org/document/10096356), [BEATs](https://ieeexplore.ieee.org/document/10887974), EAT, Dasheng |
| Backend | kNN, [kNN w/ Kmeans](https://ieeexplore.ieee.org/document/10097176), [kNN w/ SMOTE](https://ieeexplore.ieee.org/document/10889514), [kNN w/ Rescaling](https://ieeexplore.ieee.org/document/10888402), [variance-minimization-based rescaling](https://dcase.community/documents/workshop2025/proceedings/DCASE2025Workshop_Matsumoto_12.pdf), [BEAM and AdaBEAM](https://arxiv.org/abs/2603.13749) |


- Description: In this method, a frontend extracts raw features without training of meta-information label classification.
- Run Script: `jobs/asd/run/raw_*.sh`
- Recipe: `jobs/asd/recipe/raw_any.sh`
    -  This method repeats steps 2-4 for each machine type, skipping the frontend training in the first step.

```bash
# jobs/asd/recipe/raw_any.sh

experiments_extract="from_config/raw_*" # config/extract/experiments/from_config/raw_*.yaml
experiments_score="default" # config/score/experiments/default.yaml

for machine in "${machines[@]}"; do
    asdkit_extract experiments="${experiments_extract}"
    asdkit_score experiments="${experiments_score}"
    asdkit_evaluate
done
```
</details>


<details>
<summary>Pseudo labels</summary>
<br>

- Description: This method generates pseudo-labels for training the discriminative feature extractor.
- Label generation script: `special/pseudo_label/jobs/eat.sh`

[Other details](./special/pseudo_label/README.md)

</details>

## 📊 Results
We re-evaluated various ASD methods using ASDKit on multiple datasets, conducting four independent trials for each.
<br>
The recipes shown in the figure are provided in `jobs/asd/run/`.
<details>
<summary>Setups of the discriminative methods</summary>
<br>
<img src="docs/setups.png" width="400">
</details>
<br>
<img src="docs/results.png" width="800">


## Information

[Other details](./docs/details.md)
<details><summary>Third Party Code and Licenses</summary>

This project includes the following third-party components:

**BEATs**
- Source: https://github.com/microsoft/unilm/tree/master/beats
- License: MIT
- Location: `asdkit/models/pretrained_models/beats`
- Copyright (c) Microsoft Corporation

**EAT**
- Source: https://github.com/cwx-worst-one/EAT
- License: MIT
- Location: `asdkit/models/pretrained_models/eat`
- Copyright (c) 2024 Wenxi Chen
</details>


### Author

Takuya Fujimura, Nagoya University, Japan
<br>
E-mail: fujimura.takuya@g.sp.m.is.nagoya-u.ac.jp

Kevin Wilkinghoff, Aalborg University and Pioneer Centre for Artificial Intelligence, Denmark

### Citation
```
@inproceedings{fujimura2025asdkit,
    author = "Fujimura, Takuya and Wilkinghoff, Kevin and Imoto, Keisuke and Toda, Tomoki",
    title = "{ASDKit}: A Toolkit for Comprehensive Evaluation of Anomalous Sound Detection Methods",
    booktitle = "Proceedings of the 10th Workshop on Detection and Classification of Acoustic Scenes and Events (DCASE 2025)",
    address = "Barcelona, Spain",
    month = "October",
    year = "2025",
    pages = "40--44",
    isbn = "978-84-09-77652-8",
    doi = "10.5281/zenodo.17251589"
}
```
