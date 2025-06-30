cd ../..

source venv/bin/activate

# Clone fairseq
git clone git@github.com:facebookresearch/fairseq.git

# Install fairseq (this installs older versions of hydra-core and omegaconf)
pip install -e fairseq

# Install asdit (this upgrades hydra-core and omegaconf to newer versions)
pip install -e .

echo "Note: Dependency conflicts are expected but do not cause issues in our experiments."
echo " [Expected Warning 1]: fairseq 0.12.2 requires hydra-core<1.1,>=1.0.7, but you have hydra-core x.x.x."
echo " [Expected Warning 2]: fairseq 0.12.2 requires omegaconf<2.1, but you have omegaconf x.x.x."
