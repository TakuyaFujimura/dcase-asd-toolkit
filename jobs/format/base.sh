data_dir=$1
dcase=$2

cd ../..
source venv/bin/activate
asdlib-format --data_dir=$1 --dcase=$2
# python main/format.py --data_dir=$1 --dcase=$2
