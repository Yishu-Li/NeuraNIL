# Yishu Li, 2025.04.16
# This script loop through the DB and copy the listed file to new folder


DB_PATH = "/run/user/1000/gvfs/smb-share:server=files22.brown.edu,share=lrsresearch/ENG_BrainGate_Shared/DB/t17"
NIL_DATA_PATH = "/home/lys/Data/NeuraNIL/BG"
DATA_LIST = NIL_DATA_PATH + "/BlockList.csv"
EXCLUDE_BLOCKS = NIL_DATA_PATH + "/ExcludeBlocks.txt"

import csv
import os
from pathlib import Path
import shutil

def main():
    # Read the exclude blocks
    with open(EXCLUDE_BLOCKS, 'r') as f:
        exclude_blocks = f.read().splitlines()
    for i in range(len(exclude_blocks)):
        print(f'Exclude block {exclude_blocks[i]}')

    with open(DATA_LIST, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Loop through all the session day and blocks in the list
            folder_name = row['Session Day']
            block_num = row['Block No.']

            # Check if the folder exist in DB
            matches = [p for p in Path(DB_PATH).glob(f"t17.{folder_name}*") if p.is_dir()]
            matches = [folder for folder in matches if (folder / 'brainToText').exists()]
            if len(matches) == 0:
                print(f'Session day {folder_name} folder does not exit, skip.')
                continue
            elif len(matches) > 1:
                raise ValueError(f'Session day {folder_name} has multiple folders, solve manually.')
            match = matches[0]
            RedisMat = match / 'brainToText' / 'RedisMat'

            # Find the block mat file
            mat_files = [file for file in Path(RedisMat).glob(f"*({block_num}).mat")]
            for file in mat_files:
                for exclude in exclude_blocks:
                    if exclude in str(file):
                        print(f'Mat file {exclude} is excluded.')
                        mat_files.remove(file)
            if len(mat_files) == 0:
                print(f'Block {block_num} mat file from day {folder_name} does not exit, skip.')
                continue
            elif len(mat_files) > 1:
                raise ValueError(f'Block {block_num} mat file from day {folder_name} has multiple files, solve manually.')
            
            # Copy the mat file to NIL data folder
            mat_file = mat_files[0]
            mat_file_name = mat_file.name
            mat_file_path = NIL_DATA_PATH + '/' + folder_name + '/' + 'RedisMat'
            # Skip if the data already exist
            if os.path.exists(mat_file_path + '/' + mat_file_name):
                print(f'Mat file {mat_file_name} already exist, skip.')
                continue
            # Create the folder if not exist
            os.makedirs(mat_file_path, exist_ok=True)
            shutil.copy(mat_file, mat_file_path)

            # Create a readme file to record the gestures if not exist
            readme_path = NIL_DATA_PATH + '/' + folder_name + '/' + 'gestures.txt'
            with open(readme_path, 'w') as txt:
                txt.write(row['Gesture'])

if __name__ == "__main__":
    main()