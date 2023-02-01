import argparse
from pathlib import Path


def create_txt_file(experiment_folder: Path, output_file_path: Path):
    if not experiment_folder.exists():
        raise ValueError(f"The experiment path does not exist <{experiment_folder}>")
    folder_paths = sorted([str(p) + '\n' for p in experiment_folder.iterdir() if p.is_dir()])
    output_file_path.parent.mkdir(exist_ok=True, parents=True)
    with output_file_path.open('w') as f:
        f.writelines(folder_paths)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--experiment_folder', type=Path, required=True)
    parser.add_argument('-o', '--output_file_path', type=Path, required=True, help='Path to the output txt file')

    args = parser.parse_args()

    create_txt_file(**args.__dict__)
