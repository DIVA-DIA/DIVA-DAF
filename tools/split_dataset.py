import json
from pathlib import Path
from tqdm import tqdm
import numpy as np
import shutil
import argparse


def split_dataset_classification(root_folder: Path, dataset_name: str, train_split_size: int, val_split_size: int):
    originals_path = root_folder / 'originals'
    train_folder = root_folder / dataset_name / 'train'
    if train_folder.exists():
        shutil.rmtree(train_folder)
    train_folder.mkdir(parents=True)
    val_folder = root_folder / dataset_name / 'val'
    if val_folder.exists():
        shutil.rmtree(val_folder)
    val_folder.mkdir(parents=True)
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


def split_dataset_segmentation(codex_path: Path, gt_path: Path, output_path: Path, train_split_size: int,
                               val_split_size: int, extension: str):
    train_folder, val_folder = _remove_existing_folder(Path(output_path))

    possible_codex_files = np.asarray(list(codex_path.glob('*.png')))
    img_paths_reduced = np.random.choice(possible_codex_files, size=(val_split_size + train_split_size,), replace=False)
    train_idxs = np.random.choice(range(len(img_paths_reduced)), size=(train_split_size,), replace=False)
    train_mask = np.zeros(len(img_paths_reduced), dtype=bool)
    train_mask[train_idxs] = True
    val_mask = ~train_mask
    img_names = np.asarray([p.name for p in img_paths_reduced])

    # create json with split
    split_dict = {"train": img_names[train_mask].tolist(), "val": img_names[val_mask].tolist()}
    with (output_path / 'split.json').open('w') as f:
        json.dump(split_dict, f)

    # iterate through all folders
    for img_name_w_extension in img_names[train_mask]:
        # create symlink to original file
        _create_symlink_to(img_name_w_extension, source_folder_path=train_folder / 'data',
                           target_folder_path=codex_path)
        # create symlink to gt file
        _create_symlink_to(Path(img_name_w_extension).stem + extension, source_folder_path=train_folder / 'gt',
                           target_folder_path=gt_path)

    for img_name_w_extension in img_names[val_mask]:
        # create symlink to original file
        _create_symlink_to(img_name_w_extension, source_folder_path=val_folder / 'data',
                           target_folder_path=codex_path)
        # create symlink to gt file
        _create_symlink_to(Path(img_name_w_extension).stem + extension, source_folder_path=val_folder / 'gt',
                           target_folder_path=gt_path)


def _create_symlink_to(img_name_w_extension, source_folder_path: Path, target_folder_path: Path):
    source_folder_path.mkdir(parents=True, exist_ok=True)
    source_path = source_folder_path / img_name_w_extension
    source_path.symlink_to(target_folder_path / img_name_w_extension)


def _remove_existing_folder(output_path: Path):
    train_folder = output_path / 'train'
    if train_folder.exists():
        shutil.rmtree(train_folder)
    train_folder.mkdir(parents=True)
    val_folder = output_path / 'val'
    if val_folder.exists():
        shutil.rmtree(val_folder)
    val_folder.mkdir(parents=True)
    return train_folder, val_folder


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--dataset_type', type=str, required=True,
                        help='Type of the dataset (classification or segmentation)')
    parser.add_argument('-s', '--seed', type=int, required=False, default=42,
                        help='Seed for the random generator')
    intermediate_args = parser.parse_known_args()
    np.random.seed(intermediate_args[0].seed)
    if intermediate_args[0].dataset_type == 'classification':
        parser.add_argument('-r', '--root_folder', type=Path, required=True,
                            help='Path to the root folder of the dataset')
        parser.add_argument('-n', '--dataset_name', type=str, required=True,
                            help='Name of the dataset')
        parser.add_argument('-t', '--train_split_size', type=int, required=True,
                            help='Size of the training split')
        parser.add_argument('-v', '--val_split_size', type=int, required=True,
                            help='Size of the validation split')
        args = parser.parse_args()

        args_dict = args.__dict__
        del args_dict['dataset_type']
        del args_dict['seed']
        split_dataset_classification(**args_dict)

    if intermediate_args[0].dataset_type == 'segmentation':
        parser.add_argument('-c', '--codex_path', type=Path, required=True,
                            help='Path to the codex folder')
        parser.add_argument('-g', '--gt_path', type=Path, required=True,
                            help='Path to the ground truth folder')
        parser.add_argument('-o', '--output_path', type=Path, required=True,
                            help='Path to the output folder')
        parser.add_argument('-t', '--train_split_size', type=int, required=True,
                            help='Size of the training split')
        parser.add_argument('-v', '--val_split_size', type=int, required=True,
                            help='Size of the validation split')
        parser.add_argument('-e', '--extension', type=str, required=True,
                            help='File extension of the gt (e.g., .png)')

        args = parser.parse_args()
        args_dict = args.__dict__
        (args_dict['output_path'] / f"seed_{args_dict['seed']}.txt").touch()
        del args_dict['dataset_type']
        del args_dict['seed']
        split_dataset_segmentation(**args_dict)
