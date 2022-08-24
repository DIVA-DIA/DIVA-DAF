import itertools
import json
from pathlib import Path
from typing import List

import numpy as np
from tqdm import tqdm


def get_perms_with_n_fixed_positions(permutations: List, classes: List[int], matches: int) -> np.ndarray:
    perms = np.array(permutations)
    cls = np.array(classes)
    return np.array([p for p in perms if np.sum(p != cls) == matches])


if __name__ == '__main__':
    positions_w_do_not_change = 3
    classes = list(range(6))
    all_perms = list(itertools.permutations(classes))
    used_perms = get_perms_with_n_fixed_positions(all_perms, classes, positions_w_do_not_change)

    root_path = Path('/net/research-hisdoc/datasets/self-supervised/CB55/tiles_960_1344_embeded/all_files')
    perm_mapping_path = root_path / 'permutations.json'
    output_path = root_path.parent / f'{positions_w_do_not_change}_fixed_positions'
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
        for file_path in (root_path / str(perm_cls)).iterdir():
            if (class_folder / file_path.name).exists():
                continue
            (class_folder / file_path.name).symlink_to(file_path)

    json.dump([int(i) for i in permutation_classes], output_path_info.open('w'))
