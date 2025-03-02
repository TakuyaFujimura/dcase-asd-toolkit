data_dir=$1
dcase=$2

if [ -d "${data_dir}/original/${dcase}" ]; then
  echo "Data already exists. Exiting..."
  exit 1
fi

if [ "${dcase}" = "dcase2021" ]; then
  ./dcase2021.sh "${data_dir}/original"
elif [ "${dcase}" = "dcase2022" ]; then
  ./dcase2022.sh "${data_dir}/original"
elif [ "${dcase}" = "dcase2023" ]; then
  ./dcase2023.sh "${data_dir}/original"
elif [ "${dcase}" = "dcase2024" ]; then
  ./dcase2024.sh "${data_dir}/original"
else
  echo "Unknown dcase: ${dcase}"
  exit 1
fi
