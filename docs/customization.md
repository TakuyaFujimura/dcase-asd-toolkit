
# Overview of the training/testing process

- `dcase-asd-library/jobs/asd/base/base.sh` automatically executes training/testing process using `asdlib/bin/train.py` and `asdlib/bin/test.py`
- `dcase-asd-library/jobs/asd/example/run.sh` is an wrapper script of `base.sh`
- Please refer `run.sh`. It includes an explanation of which configuration file will be used

![w:1000 center](./overview.drawio.png)


# Label

- Several basic labels are provided in `dcase-asd-library/labels`.
- You can also create your own label (See `asdlib/bin/label.py`, `config/label`, and `jobs/labeling/base.sh`)

![w:1000 center](./label.drawio.png)
