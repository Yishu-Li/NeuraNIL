# Activate the conda environment before running the script
# This is just an example!!!

# Default LSTM
python main.py --epochs=100 --dataset.data="BG" --dataset.random_split=True --days_exclude='[0, 2, 3, 4, 5, 6, 7, 8, 9]' --lstm.pad=False --lstm.activation='softmax' --lstm.norm=True


# MLP
python main.py --epochs=70 --dataset.data="BG" --dataset.random_split=True --dataset.days_exclude='[0, 2, 3, 4, 5, 6, 7, 8, 9]' --model="MLP" --mlp.activation="softmax"


# LDA & GNB
python main.py --epochs=70 --dataset.data="BG" --dataset.random_split=True --dataset.days_exclude='[0, 2, 3, 4, 5, 6, 7, 8, 9]' --model="LDA" --lda.n_components=2
python main.py --epochs=70 --dataset.data="BG" --dataset.random_split=True --dataset.days_exclude='[0, 2, 3, 4, 5, 6, 7, 8, 9]' --model="GNB"