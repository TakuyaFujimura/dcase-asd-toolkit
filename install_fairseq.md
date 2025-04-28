```
[dcase-asd-toolkit]$ git clone git@github.com:facebookresearch/fairseq.git
[dcase-asd-toolkit]$ source venv/bin/activate
(venv) [dcase-asd-toolkit]$ pip install -e fairseq`
# Here, fairseq installs old versions of hydra-core and omegaconf
(venv) [dcase-asd-toolkit]$ pip install -e .
# Here, asdit installs new versions of hydra-core and omegaconf
# In our experiments, mismatch of these versions does not cause any problems.
# The following error message appears
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
fairseq 0.12.2 requires hydra-core<1.1,>=1.0.7, but you have hydra-core 1.2.0 which is incompatible.
fairseq 0.12.2 requires omegaconf<2.1, but you have omegaconf 2.2.3 which is incompatible.
