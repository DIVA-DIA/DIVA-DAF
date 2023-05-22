import json
from pathlib import Path
from tqdm import tqdm

if __name__ == '__main__':
    originals_codex_path = Path(
        '/net/research-hisdoc/datasets/semantic_segmentation/datasets/CB55-splits/originals/codex')
    originals_gt_path = Path(
        '/net/research-hisdoc/datasets/semantic_segmentation/datasets/CB55-splits/originals/truthL')
    output_root_path = Path('/net/research-hisdoc/datasets/semantic_segmentation/datasets/CB55-splits/split3')

    for split_name in ['training', 'validation', 'test']:
        json_path_train = output_root_path / f'content_{split_name}.json'
        train_files = json.load(json_path_train.open('r'))
        for file_name in tqdm(train_files):
            file_path = originals_codex_path / (file_name + '.png')
            gt_path = originals_gt_path / (file_name + '.gif')
            train_path = output_root_path / split_name
            train_path_gt = train_path / 'gt'
            train_path_data = train_path / 'data'
            train_path_data.mkdir(parents=True, exist_ok=True)
            train_path_gt.mkdir(parents=True, exist_ok=True)
            train_path.mkdir(parents=True, exist_ok=True)
            (train_path_data / (file_name + '.png')).symlink_to(file_path)
            (train_path_gt / (file_name + '.gif')).symlink_to(gt_path)
