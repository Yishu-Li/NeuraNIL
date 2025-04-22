# Activate the conda environment before running the script
# This is just an example!!!

python main.py --epochs=20 --dataset.data="BG" --dataset.random_split=True --dataset.labels_exclude='[4, 5]' --lstm.pad=True --lstm.activation='softmax'