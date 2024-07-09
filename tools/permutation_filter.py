import argparse
import itertools
import json
from pathlib import Path
from typing import List

import numpy as np
from tqdm import tqdm


def get_perms_with_n_fixed_positions(permutations: List, classes: List[int], matches: int) -> np.ndarray:
    perms = np.array(permutations)
    cls = np.array(classes)
    return np.array([p for p in perms if np.sum(p == cls) == matches])


def filter_by_permutations(number_of_tiles: int, pos_not_change: int, input_path: Path) -> None:
    classes = list(range(number_of_tiles))
    all_perms = list(itertools.permutations(classes))
    used_perms = get_perms_with_n_fixed_positions(all_perms, classes, pos_not_change)
    perm_mapping_path = input_path / 'permutations.json'
    output_path = input_path.parent / f'{pos_not_change}_fixed_positions'
    output_path_info = output_path / 'permutations.json'
    output_path.mkdir(exist_ok=True)
    perm_mappings = np.asarray(json.load(perm_mapping_path.open()))
    permutation_classes = [np.where(np.all(perm_mappings == tuple(p), axis=1))[0][0] for p in used_perms if
                           # checks if there is a match in the permutation mapping
                           np.any(np.all(perm_mappings == tuple(p), axis=1))]
    assert len(used_perms) == len(permutation_classes)
    for i, perm_cls in tqdm(enumerate(permutation_classes)):
        class_folder = output_path / str(i)
        class_folder.mkdir(exist_ok=True)
        for file_path in (input_path / str(perm_cls)).iterdir():
            if (class_folder / file_path.name).exists():
                continue
            (class_folder / file_path.name).symlink_to(file_path)
    json.dump([int(i) for i in permutation_classes], output_path_info.open('w'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--pos_not_change',
                        help="Amount of positions that do not change. Hence, are at the correct position",
                        type=int, default=3)
    parser.add_argument('-i', '--input_path',
                        help="Path to all generated permutations.",
                        type=Path, required=True)
    parser.add_argument('-n', '--number_of_tiles',
                        help="Number of tiles that exists in one image",
                        type=Path, required=True)

    args = parser.parse_args()
    # positions_w_do_not_change = 2
    # input_path = Path('/net/research-hisdoc/datasets/self-supervised/CB55/tiles_960_1344_embeded/all_files')
    filter_by_permutations(**args.__dict__)
