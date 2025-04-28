# Activate the conda environment before running the script
# This is just an example!!!

# Default LSTM
python main.py --epochs=100 --dataset.data="BG" --dataset.split_method="random" --days_exclude='[0, 2, 3, 4, 5, 6, 7, 8, 9]' --lstm.pad=False --lstm.activation='softmax' --lstm.norm=True
python main.py --epochs=100 --dataset.data="BG" --dataset.split_method="random" --days_exclude='[0, 2, 3, 4, 5, 6, 7, 8, 9]' --lstm.pad=False --lstm.activation='softmax' --lstm.norm=True --lstm.ifconv=True --lstm.num_layer=1
python main.py --dataset.data="BG" --dataset.split_method="day" --lstm.pad=False --lstm.activation='softmax' --lstm.pad=False --lstm.activation='softmax' --lstm.norm=True --lstm.ifconv=True --lstm.num_layer=1 --dataset.train_days='[0, 1, 2, 3, 4, 5, 6]' --dataset.test_days='[7, 8, 9]'

# MLP
python main.py --epochs=70 --dataset.data="BG" --dataset.split_method="random" --dataset.days_exclude='[0, 2, 3, 4, 5, 6, 7, 8, 9]' --model="MLP" --mlp.activation="softmax"


# LDA & GNB
python main.py --epochs=70 --dataset.data="BG" --dataset.split_method="random" --dataset.days_exclude='[0, 2, 3, 4, 5, 6, 7, 8, 9]' --model="LDA" --lda.n_components=2
python main.py --epochs=70 --dataset.data="BG" --dataset.split_method="random" --dataset.days_exclude='[0, 2, 3, 4, 5, 6, 7, 8, 9]' --model="GNB"

# NeuraNIL
python main.py --dataset.data="BG" --dataset.split_method="day" --days_exclude='[0, 2, 3, 4, 5, 6, 7, 8, 9]' --lstm.pad=False --lstm.activation='softmax' --lstm.norm=True --lstm.ifconv=True --lstm.num_layer=2 --model='NeuraNIL' --meta.learner='LSTM' --meta.classifier='MLP'
python main.py --dataset.data="BG" --dataset.split_method="day" --lstm.pad=False --lstm.activation='softmax' --lstm.norm=True --lstm.ifconv=True --lstm.num_layer=2 --model='NeuraNIL' --meta.learner='LSTM' --meta.classifier='MLP' --dataset.train_days='[0, 1, 2, 3, 4, 5, 6]' --dataset.test_days='[7, 8, 9]'