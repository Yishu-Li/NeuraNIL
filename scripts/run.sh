# Activate the conda environment before running the script
# This is just an example!!!


python main.py --epochs=5 --dataset.data="BG" --dataset.random_split=True --lstm.pad=True --lstm.activation='softmax'

python main.py --epochs=20 --dataset.data="BG" --dataset.random_split=True --dataset.labels_exclude='[4, 5]' --dataset.days_exclude='[0, 1, 2, 3, 4, 5, 6, 7, 8]' --lstm.pad=True --lstm.activation='softmax'