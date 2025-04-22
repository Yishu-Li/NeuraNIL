# Yishu Li, 2025.04.16
# This script is used to preprocess the BrainGate data for NeuraNIL

NIL_DATA_PATH = "/home/lys/Data/NeuraNIL/BG"
MAPPING_ALL = ['Fist', 'No', 'C', 'Sign8', 'LThumb', 'DO_NOTHING']

import os
from pathlib import Path
import numpy as np
from bg_utils import formatSessionData

def main():
    # Get the session day data
    session_day_folders = [p for p in os.listdir(NIL_DATA_PATH) if os.path.isdir(os.path.join(NIL_DATA_PATH, p))]
    print(f'Found {len(session_day_folders)} session day folders:')
    print(session_day_folders)

    # Set save directory
    save_dir = "Data/BG"

    # Record the number of all trials
    count = {MAPPING_ALL[i]: 0 for i in range(len(MAPPING_ALL))}

    # Loop through each session day folder
    for folder in session_day_folders:
        print('=' * 50)
        dataDir = str(Path(NIL_DATA_PATH, folder))

        # Get the block numbers
        RedisMat = str(Path(dataDir, 'RedisMat'))
        blocks = sorted([int(x.split('(')[1].split(')')[0]) for x in os.listdir(RedisMat) if x.endswith('.mat')])
        print(f'Found {len(blocks)} blocks in {folder}:')
        print(f"Block Nums: {blocks}")

        # Get the mapping from gesture.txt file
        mapping_file = str(Path(dataDir, 'gestures.txt'))
        mapping = []
        with open(mapping_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                mapping.append(line.strip().split()[1])
        mapping.append('DO_NOTHING')
        print(f"Mapping: {mapping}")

        # Read the data
        data = formatSessionData(
                            blocks,
                            trialsToRemove=[],
                            dataDir=dataDir,
                            channels_to_exclude=[],
                            channels_to_zero=[],
                            includeThreshCrossings=True,
                            includeSpikePower=True,
                            spikePowerMax=10000,
                            globalStd=True,
                            zscoreData=True,
                            bin_compression_factor=2)
        
        inputFeatures = data['inputFeatures']
        cues = data['transcriptions']
        if len(inputFeatures) != len(cues):
            raise ValueError(f'Input features and cues length mismatch: {len(inputFeatures)} != {len(cues)}')
        cues = [cue.strip() for cue in cues]
        # Change the cues to the mapping
        # cues = np.array([mapping[int(cue)-1] for cue in cues if cue.isdigit()])
        for i, cue in enumerate(cues):
            if cue.isdigit():
                cues[i] = mapping[int(cue)-1]
        # Change the cues to the global mapping index and add the features
        global_cues = []
        features = []
        minsz = np.shape(inputFeatures[0])[0]  # Make sure all input features are the same size
        for x in inputFeatures:
            # print(np.shape(x)[0])
            if np.shape(x)[0] < minsz:
                minsz = np.shape(x)[0]
        for i in range(len(cues)):
            if cues[i] in MAPPING_ALL:
                global_cues.append(MAPPING_ALL.index(cues[i]))
                features.append(inputFeatures[i][0:minsz, :])
        global_cues = np.array(global_cues)
        features = np.array(features)
        
        # Check if the feature and cues length match
        if global_cues.shape[0] != features.shape[0]:
            raise ValueError(f'Global cues and features length mismatch: {global_cues.shape[0]} != {features.shape[0]}')
        
        # Count the number of trials for each gesture
        for i in range(len(global_cues)):
            count[MAPPING_ALL[global_cues[i]]] += 1
        
        # Save the data
        save_path = os.path.join(save_dir, folder)
        if os.path.exists(save_path):
            print(f'Save path {save_path} already exists, skip.')
        else:
            print(f'Saving data to {save_path}')
            os.makedirs(save_path, exist_ok=True)
            np.savez(os.path.join(save_path, 'data.npz'), X=features, y=global_cues)

    # Print the number of trials for each gesture
    print('Number of trials for each gesture:')
    for i in range(len(MAPPING_ALL)):
        print(f'{MAPPING_ALL[i]}: {count[MAPPING_ALL[i]]}')

            


if __name__ == "__main__":
    main()