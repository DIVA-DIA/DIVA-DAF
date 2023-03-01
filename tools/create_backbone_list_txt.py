import argparse
from pathlib import Path


def create_txt_file(experiment_folder: Path, output_file_path: Path):
    if not experiment_folder.exists():
        raise ValueError(f"The experiment path does not exist <{experiment_folder}>")
    best_paths_backbone = []
    best_paths_header = []
    for p in experiment_folder.iterdir():
        checkpoint_dir_path = p / 'checkpoints'
        for c in checkpoint_dir_path.iterdir():
            if c.is_dir():
                best_paths_backbone.append(c / 'backbone.pth')
                best_paths_header.append(c / 'header.pth')

    checkpoint_paths_bb = sorted([str(p) for p in best_paths_backbone if p.is_file()])
    checkpoint_paths_header = sorted([str(p) for p in best_paths_header if p.is_file()])

    with output_file_path.open('w') as f:
        f.write('weights_backbone=(')
        for b in checkpoint_paths_bb:
            f.write(f'"{b}"\n')
        f.write(')')
        f.write("\n\n")
        f.write('weights_header=(')
        for b in checkpoint_paths_header:
            f.write(f'"{b}"\n')
        f.write(')')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--experiment_folder', type=Path, required=True)
    parser.add_argument('-o', '--output_file_path', type=Path, required=True, help='Path to the output txt file')

    args = parser.parse_args()

    create_txt_file(**args.__dict__)
