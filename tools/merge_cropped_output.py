from collections import defaultdict
from pathlib import Path

import re
import numpy as np

from src.datamodules.hisDBDataModule.DIVAHisDBDataModule import DIVAHisDBDataModuleCropped
from src.datamodules.hisDBDataModule.cropped_hisdb_dataset import CroppedHisDBDataset
from src.models.semantic_segmentation.utils.output_tools import merge_patches, save_output_page_image


def merge_cropped_output(data_dir: Path, prediction_path: Path, outdir: Path):
    data_module = DIVAHisDBDataModuleCropped(data_dir=data_dir)
    num_classes = data_module.num_classes
    class_encodings = data_module.class_encodings

    img_paths_per_page = CroppedHisDBDataset.get_gt_data_paths_cropped(directory=data_dir / 'test')

    dataset_img_name_list = []
    dataset_dict = defaultdict(list)
    for img_path, gt_path, img_name, patch_name, (x, y) in img_paths_per_page:
        if img_name not in dataset_img_name_list:
            dataset_img_name_list.append(img_name)
        dataset_dict[img_name].append([img_path, gt_path, patch_name, x, y])

    dataset_img_name_list = sorted(dataset_img_name_list)

    # sort dataset_dict lists
    for img_name in dataset_dict.keys():
        dataset_dict[img_name] = sorted(dataset_dict[img_name], key=lambda v: (v[4], v[3]))

    # Merge prediction patches on canvas
    img_name_list = sorted([str(n.name) for n in prediction_path.iterdir() if n.is_dir()])

    # check if all images from the dataset are found in the prediction output
    assert sorted(dataset_img_name_list) == sorted(img_name_list)

    for img_name in img_name_list:
        patches_folder = prediction_path / img_name
        coordinates = re.compile(r'.+_x(\d+)_y(\d+)\.npy$')

        if not patches_folder.is_dir():
            continue

        patches_list = []
        for patch_file in patches_folder.glob(f'{img_name}*.npy'):
            m = coordinates.match(patch_file.name)
            if m is None:
                continue
            x = int(m.group(1))
            y = int(m.group(2))
            patch = np.load(str(patch_file))
            patches_list.append((patch, x, y))
        patches_list = sorted(patches_list, key=lambda v: (v[2], v[1]))

        # Create new canvas
        canvas_width = patches_list[-1][0].shape[1] + patches_list[-1][1]
        canvas_height = patches_list[-1][0].shape[2] + patches_list[-1][2]
        canvas_size = (num_classes,
                       canvas_width,
                       canvas_height)
        canvas = np.empty(canvas_size)
        canvas.fill(np.nan)

        for patch, x, y in patches_list:
            # Add the patch to the image
            canvas = merge_patches(patch, (x, y), canvas)

        # Save the image when done
        if not np.isnan(np.sum(canvas)):
            # Save the final image (image_name, output_image, output_folder, class_encoding)
            save_output_page_image(image_name=f'{img_name}.png', output_image=canvas,
                                   output_folder=outdir / 'pred',
                                   class_encoding=class_encodings)
        else:
            print(f'WARNING: Test image {img_name} was not written! It still contains NaN values.')


if __name__ == '__main__':
    merge_cropped_output(
        data_dir=Path('/data/usl_experiments/semantic_segmentation/datasets_cropped/CB55-10-segmentation'),
        prediction_path=Path('/home/paul/unsupervised_learning/logs/runs/2021-08-23/17-29-08/test_images/patches'),
        outdir=Path('/home/paul/unsupervised_learning/logs/runs/2021-08-23/17-29-08/test_images/result'),

    )
