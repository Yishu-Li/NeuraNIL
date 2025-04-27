#!/usr/bin/env python3
"""
falcon_preprocess.py

Extract and save raw arrays from each FALCON session:
 - For .nwb files: uses the official loader if available.
 - For .mat files: uses scipy.io.loadmat.

Output: processed_data/{session}.pt ONLY 4 WORK 

Usage:
  conda activate csci1470
  pip install falcon-challenge torch scipy numpy
  python falcon_preprocess.py --data_dir . --output_dir processed_data
"""

import os, argparse
import numpy as np
import scipy.io
import torch
from pathlib import Path

# Try to import the official loader for NWB
try:
    from falcon_challenge.dataloaders import load_nwb
    from falcon_challenge.config    import FalconTask
    HAVE_OFFICIAL = True
except ImportError:
    HAVE_OFFICIAL = False

# Map folder→task for official loader
TASK_MAP = {
    '000941': FalconTask.m1 if HAVE_OFFICIAL else None,
    '000953': FalconTask.m2 if HAVE_OFFICIAL else None,
    '000954': FalconTask.h1 if HAVE_OFFICIAL else None,
    '000950': FalconTask.h2 if HAVE_OFFICIAL else None,
    '001046': FalconTask.b1 if HAVE_OFFICIAL else None,
}

SESSIONS = list(TASK_MAP.keys())

def find_file(folder: Path, exts):
    """Return first file under folder ending in any of exts."""
    for r, _, files in os.walk(folder):
        for f in files:
            if f.lower().split('.')[-1] in exts:
                return Path(r)/f
    return None

def load_mat(path: Path):
    """Load .mat → spike_counts, labels, time_stamps."""
    md = scipy.io.loadmat(str(path), squeeze_me=True)
    return md['spike_counts'], md['labels'], md['time_stamps']

def preprocess_session(session: str, data_dir: Path, out_dir: Path):
    folder = data_dir / session
    if not folder.is_dir():
        print(f"[{session}] Folder missing; skipping.")
        return

    # Attempt official NWB loader
    if HAVE_OFFICIAL:
        nwb = find_file(folder, ('nwb',))
        if nwb:
            task = TASK_MAP[session]
            print(f"[{session}] Loading NWB via official loader: {nwb.name}")
            neural, behav, trial_change, eval_mask = load_nwb(str(nwb), task)
            # Convert to numpy/torch as appropriate
            neural = np.asarray(neural)
            # behav may be numeric array or list of strings
            try:
                behav_arr = np.asarray(behav)
                behav_to_save = behav_arr
            except:
                behav_to_save = list(behav)
            tc = torch.from_numpy(trial_change.astype(bool))
            em = torch.from_numpy(eval_mask.astype(bool))

            out = out_dir / f"{session}.pt"
            torch.save({
                'neural':       torch.from_numpy(neural).float(),
                'behav':        behav_to_save,
                'trial_change': tc,
                'eval_mask':    em,
            }, str(out))
            print(f"[{session}] Saved → {out}\n")
            return

    ### mat document amnagement 
    mat = find_file(folder, ('mat',))
    if mat:
        print(f"[{session}] Loading MAT: {mat.name}")
        spikes, labels, times = load_mat(mat)
        out = out_dir / f"{session}.pt"
        torch.save({
            'spike_counts': torch.from_numpy(spikes).float(),
            'labels':       torch.from_numpy(np.asarray(labels)).long(),
            'time_stamps':  torch.from_numpy(times).float(),
        }, str(out))
        print(f"[{session}] Saved → {out}\n")
        return

    # Nothing found
    print(f"[{session}] BAD No .nwb or .mat found under {folder}")

def main(data_dir: str, output_dir: str):
    data_dir = Path(data_dir)
    out_dir  = Path(output_dir)
    out_dir.mkdir(exist_ok=True)
    for sess in SESSIONS:
        preprocess_session(sess, data_dir, out_dir)
    print("GOOD Preprocessing complete.")

if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",   required=True,
                   help="Directory containing 000941…001046 folders")
    p.add_argument("--output_dir", required=True,
                   help="Where to save processed .pt files")
    args = p.parse_args()
    main(args.data_dir, args.output_dir)
