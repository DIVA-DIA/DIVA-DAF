from pathlib import Path
from tqdm import tqdm
import numpy as np
import shutil
import argparse


def split_dataset(root_folder: Path, dataset_name: str, train_split_size: int, val_split_size: int):
    originals_path = root_folder / 'originals'
    train_folder = root_folder / dataset_name / 'train'
    if train_folder.exists():
        shutil.rmtree(train_folder)
    train_folder.mkdir()
    val_folder = root_folder / dataset_name / 'val'
    if val_folder.exists():
        shutil.rmtree(val_folder)
    val_folder.mkdir()
    possible_files = np.asarray(list(next(originals_path.iterdir()).glob('*.png')))
    img_paths_reduced = np.random.choice(possible_files, size=(val_split_size + train_split_size,), replace=False)
    train_idxs = np.random.choice(range(len(img_paths_reduced)), size=(train_split_size,), replace=False)
    train_mask = np.zeros(len(img_paths_reduced), dtype=bool)
    train_mask[train_idxs] = True
    val_mask = ~train_mask
    img_names = np.asarray([p.name for p in img_paths_reduced])
    # iterate through all folders
    for folder in tqdm(originals_path.iterdir()):
        if folder.is_file():
            continue
        perm_index = folder.name
        for train_file_name in img_names[train_mask]:
            # create symlink to original file and save into train folder
            source_path = train_folder / str(perm_index) / train_file_name
            source_path.parent.mkdir(parents=True, exist_ok=True)
            if source_path.is_symlink():
                continue
            source_path.symlink_to(originals_path / str(perm_index) / train_file_name)

        for val_file_name in img_names[val_mask]:
            # create symlink to original file and save into val folder
            source_path = val_folder / str(perm_index) / val_file_name
            source_path.parent.mkdir(parents=True, exist_ok=True)
            if source_path.is_symlink():
                continue
            source_path.symlink_to(originals_path / str(perm_index) / val_file_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('r', '--root_folder', type=str, required=True,
                        help='Root folder of the dataset. Where the originals folder is in')
    parser.add_argument('-n', '--dataset_name', type=str, required=True,
                        help='Name of the datase folder in the root folder')
    parser.add_argument('-t', '--train_split_size', type=int, required=True, help='Size of the train split')
    parser.add_argument('-v', '--val_split_size', type=int, required=True, help='Size of the val split')
    args = parser.parse_args()

    split_dataset(**args.__dict__)
