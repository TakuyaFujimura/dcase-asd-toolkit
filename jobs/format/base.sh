data_dir=$1
dcase=$2


cd ../../..
source venv/bin/activate
python main/format.py --data_dir=$1 --dcase=$2
