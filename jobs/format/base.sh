data_dir=$1
dcase=$2

cd ../..
source venv/bin/activate
python -m asdlib.bin.format --data_dir=$1 --dcase=$2
