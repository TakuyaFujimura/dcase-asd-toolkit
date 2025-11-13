cd ..

config_name=raw_eat_kmeans_8
for dcase in dcase2023; do
    python main.py --config-name=${config_name} dcase=${dcase}
    python main.py --config-name=${config_name} dcase=${dcase} machinewise=false output_label_name=${config_name}_all
done
