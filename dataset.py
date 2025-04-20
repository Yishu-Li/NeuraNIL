import torch
from pathlib import Path
import numpy as np
import dataclasses

def find_session_dirs(dataset, excludeSession=None):
    """
    Find all session directories in the given session directories.
    """

    use_session_dirs = []
    if dataset == "BG":
        session_dirs = Path("./Data/BG")
    elif dataset == "FALCON":
        session_dirs = Path("./Data/FALCON")
    
    # Iterate through all session directories
    for ses_dir in session_dirs.iterdir():
        if ses_dir.is_dir():
            excludeFlag = 0

            # Exclude
            if excludeSession is not None:
                for exSes in excludeSession:
                    if exSes in ses_dir.name:
                        excludeFlag = 1
                        break
            

            if not excludeFlag:
                use_session_dirs.append(ses_dir)
    
    # Sort the block directories
    use_session_dirs.sort()

    return use_session_dirs


def load_data_with_daylabels(dataset):
    """
    Load data with day labels.
    """
    session_dirs = find_session_dirs(dataset)
    data = []
    labels = []
    day_labels = []
    for ses_dir in session_dirs:
        # Load the data from the session directory
        data.append(np.load(ses_dir / "data.npz")['X'])
        labels.append(np.load(ses_dir / "data.npz")['y'])
        day_labels.append([ses_dir.name] * len(labels[-1]))

        # Print the number of trials in each session
        print(f"Session {ses_dir.name} has {len(labels[-1])} trials.")

    return data, labels, day_labels










if __name__ == "__main__":
    # For debugging only
    dataset = "BG"
    use_session_dirs = find_session_dirs(dataset)
    print("Session directories:", use_session_dirs)

    data, labels, day_labels = load_data_with_daylabels(dataset)
    print("Data shape:", len(data))
    print("Labels shape:", len(labels))
    print("Day labels shape:", len(day_labels))