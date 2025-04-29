
import os
import numpy as np
import scipy.io
from pathlib import Path


SESSIONS = ["t5.2019.05.08","t5.2019.11.25","t5.2019.12.09","t5.2019.12.11","t5.2019.12.18","t5.2019.12.20","t5.2020.01.06","t5.2020.01.08","t5.2020.01.13","t5.2020.01.15"]
MAPPING = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

ROOT = Path(__file__).parent.parent.parent
RAW  = ROOT / "Data" / "FALCON"
OUT  = ROOT / "Data" / "FALCON"

def load_cube(matfile, key):
    return scipy.io.loadmat(str(matfile), squeeze_me=True)[key]

def preprocess_session(sess, day):
    inf = RAW / sess
    outf = OUT / sess
    outf.mkdir(parents=True, exist_ok=True)

    ### concatenate fils into 1 dataset
    # cube = load_cube(inf/"singleLetters.mat", "neuralActivityCube")
    # labels = load_cube(inf/"singleLetters.mat", "characterCues")
    # if (inf/"sentences.mat").exists():
    #     cube2 = load_cube(inf/"sentences.mat", "neuralActivityCube")
    #     lab2  = load_cube(inf/"sentences.mat", "intendedText")
    #     cube  = np.concatenate([cube, cube2], axis=0)
    #     labels = np.concatenate([labels, lab2], axis=0)
    all_cubes = []
    all_labels = []
    all_day_labels = []
    for i, letter in enumerate(MAPPING):
        key = f"neuralActivityCube_{letter}"
        cube = load_cube(inf/f"singleLetters.mat", key)
        labels = np.full(cube.shape[0], i)
        day_labels = np.full(cube.shape[0], day)

        all_cubes.append(cube)
        all_labels.append(labels)
        all_day_labels.append(day_labels)

    cube = np.concatenate(all_cubes, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    day_labels = np.concatenate(all_day_labels, axis=0)
    print(f"[{sess}] → cube shape: {cube.shape}, labels shape: {labels.shape}, day_labels shape: {day_labels.shape}")
  
    # z-score the data
    num_trials, num_bins, num_channels = cube.shape
    cube = cube.reshape(-1, num_channels)
    if cube.shape[1] == 0:
        raise ValueError("The neural activity cube is empty.")
    cube = cube - np.mean(cube, axis=0)
    cube = cube / np.std(cube, axis=0)
    cube = cube.reshape(num_trials, num_bins, num_channels)

    X = cube.astype(np.float32)         ###keeps up with shape
    y = labels.astype(np.int32)
    day_labels = day_labels.astype(np.int32) 
    np.savez(outf/"data.npz", X=X, y=y, day_labels=day_labels)
    print(f"[{sess}] → {X.shape[0]} trials, saved to {outf}")

def main():
    print("ROOT:", ROOT)
    print("RAW:", RAW)
    print("OUT:", OUT)
    for day, s in enumerate(SESSIONS):
        preprocess_session(s, day)
    print(" All sessions have been preprocessed into the Data/FALCON/*/data.npz")

if __name__=="__main__":
    main()
