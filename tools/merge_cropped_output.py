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

    # Merge prediction patches on canvas
    img_name_list = [str(n.name) for n in prediction_path.iterdir() if n.is_dir()]
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
        canvas_size = (num_classes,
                       patches_list[-1][0].shape[1] + patches_list[-1][1],
                       patches_list[-1][0].shape[2] + patches_list[-1][2])
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
        prediction_path=Path('/data/usl_experiments/tmp_testing_output/baby_unet_cropped_cb55_v2021_04_22a/patches'),
        outdir=Path('/data/usl_experiments/tmp_testing_output/baby_unet_cropped_cb55_v2021_04_22a/result'),

    )
