# This script requires that the embeddings have already been extracted by jobs/asd/call/raw_eat.sh

source ../../../venv/bin/activate

cd ..

feature=raw_eat
config_name=kmeans_8
for dcase in dcase2023 dcase2024; do
    recipe_dir=${feature}/0/output/last
    python main.py --config-name=${config_name} dcase=${dcase} recipe_dir=${recipe_dir} output_label_name=${feature}_${config_name}
    python main.py --config-name=${config_name} dcase=${dcase} recipe_dir=${recipe_dir} output_label_name=${feature}_${config_name}_all machinewise=false 
done
