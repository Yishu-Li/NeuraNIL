
"""
Split continuous FALCON H2 data into trials, pad properly,
save clean Data/FALCON/H2/data.npz with X=(30, <trial_len>, features), y=(30,)
"""

import torch
import numpy as np
from pathlib import Path

PT_PATH = Path(__file__).parent / "000950.pt"
OUT_DIR = Path(__file__).parent.parent.parent / "Data" / "FALCON" / "H2"

def main():
    data = torch.load(PT_PATH)
    neural = data["neural"]              
    trial_change = data["trial_change"]   
    behav = data["behav"]               

    neural = neural.numpy()
    trial_change = trial_change.numpy().astype(bool)

  
    start_idxs = np.flatnonzero(trial_change)
    end_idxs = np.concatenate([start_idxs[1:], [neural.shape[0]]])

   
    X_trials = []
    for start, end in zip(start_idxs, end_idxs):
        trial = neural[start:end]
        X_trials.append(trial)

   
    max_len = max(trial.shape[0] for trial in X_trials)
    n_trials = len(X_trials)
    n_features = neural.shape[1]

    X_fixed = np.zeros((n_trials, max_len, n_features), dtype=np.float32)
    for i, trial in enumerate(X_trials):
        length = trial.shape[0]
        X_fixed[i, :length, :] = trial

    # Prepare labels
    y = np.array([int(np.asarray(label).flatten()[0]) for label in behav], dtype=int)

    # Save
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(OUT_DIR / "data.npz", X=X_fixed, y=y)

    print(f"Done, saved 2 {OUT_DIR/'data.npz'}")
    print(f"  X shape: {X_fixed.shape}, y shape: {y.shape}")

if __name__ == "__main__":
    main()
