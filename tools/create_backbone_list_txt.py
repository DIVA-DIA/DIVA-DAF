import argparse
from pathlib import Path


def create_txt_file(experiment_folder: Path, output_file_path: Path):
    if not experiment_folder.exists():
        raise ValueError(f"The experiment path does not exist <{experiment_folder}>")
    best_paths = []
    for p in experiment_folder.iterdir():
        checkpoint_dir_path = p / 'checkpoints'
        for c in checkpoint_dir_path.iterdir():
            if c.is_dir():
                best_paths.append(c / 'backbone.pth')

    checkpoint_paths = sorted([str(p) for p in best_paths if p.is_file()])
    with output_file_path.open('w') as f:
        f.write('weights=(')
        for l in checkpoint_paths:
            f.write(f'"{l}"\n')
        f.write(')')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--experiment_folder', type=Path, required=True)
    parser.add_argument('-o', '--output_file_path', type=Path, required=True, help='Path to the output txt file')

    args = parser.parse_args()

    create_txt_file(**args.__dict__)
