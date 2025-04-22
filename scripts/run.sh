# Activate the conda environment before running the script
# This is just an example!!!

# Default LSTM
python main.py --epochs=20 --dataset.data="BG" --dataset.random_split=True --dataset.labels_exclude='[4, 5]' --lstm.pad=True --lstm.activation='softmax'


# MLP
python main.py --epochs=70 --dataset.data="BG" --dataset.random_split=True --dataset.days_exclude='[0, 2, 3, 4, 5, 6, 7, 8, 9]' --model="MLP" --mlp.activation="softmax"