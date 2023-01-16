import json
from pathlib import Path
from tqdm import tqdm

if __name__ == '__main__':
    originals_codex_path = Path(
        '/net/research-hisdoc/datasets/self-supervised/CB55/resized/960_1344/filtered/')
    originals_gt_path = Path(
        '/net/research-hisdoc/datasets/self-supervised/CB55/binary_cleaned/sauvola/savola_rlsa_25_median_5')
    output_root_path = Path('/net/research-hisdoc/datasets/self-supervised/CB55/binary_cleaned/sauvola/dataset_rlsa_1')
    json_path = Path("/net/research-hisdoc/datasets/self-supervised/CB55/binary_cleaned/sauvola/dataset_1/split.json")

    split_names = json.load(json_path.open('r'))
    for split in ['train', 'val']:
        for file_name in tqdm(split_names[split]):
            file_path = originals_codex_path / (file_name)
            gt_path = originals_gt_path / (file_name)
            split_path = output_root_path / split
            split_path_gt = split_path / 'gt'
            split_path_data = split_path / 'data'
            split_path_data.mkdir(parents=True, exist_ok=True)
            split_path_gt.mkdir(parents=True, exist_ok=True)
            split_path.mkdir(parents=True, exist_ok=True)
            (split_path_data / (file_name)).symlink_to(file_path)
            (split_path_gt / (file_name)).symlink_to(gt_path)
