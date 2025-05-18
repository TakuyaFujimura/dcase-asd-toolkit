
# Overview of the training/testing process

- `dcase-asd-toolkit/jobs/asd/base/base.sh` automatically executes training/testing process using `asdit/bin/train.py` and `asdit/bin/test.py`
- `dcase-asd-toolkit/jobs/asd/example/run.sh` is an wrapper script of `base.sh`
- Please refer `run.sh`. It includes an explanation of which configuration file will be used

![w:1000 center](./overview.drawio.png)


# Label

- Several basic labels are provided in `dcase-asd-toolkit/labels`.
- You can also create your own label (See `asdit/bin/label.py`, `config/label`, and `jobs/labeling/base.sh`)

![w:1000 center](./label.drawio.png)
