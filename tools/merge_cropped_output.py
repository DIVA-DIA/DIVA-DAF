import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

from src.datamodules.hisDBDataModule.DIVAHisDBDataModule import DIVAHisDBDataModuleCropped
from src.datamodules.datasets.cropped_hisdb_dataset import CroppedHisDBDataset
from src.tasks.semantic_segmentation.utils.output_tools import merge_patches, save_output_page_image
from tools.generate_cropped_dataset import pil_loader
from tools.viz import visualize


@dataclass
class CropData:
    name: Path
    offset_x: int
    offset_y: int
    height: int
    width: int
    pred_path: Path
    img_path: Path
    gt_path: Path


def merge_cropped_output(datamodule_path, prediction_path: Path, output_path: Path):
    info_list = ['Running merge_cropped_output.py:',
                 f'- start_time:        \t{datetime.now():%Y-%m-%d_%H-%M-%S}',
                 f'- datamodule_path:   \t{datamodule_path}',
                 f'- prediction_path:   \t{prediction_path}',
                 f'- output_path:       \t{output_path}',
                 '']  # empty string to get linebreak at the end when using join
    info_str = '\n'.join(info_list)
    print(info_str)

    # Write info_cropped_dataset.txt
    output_path.mkdir(parents=True, exist_ok=True)
    info_file = output_path / 'info_merge_cropped_output.txt'
    with info_file.open('a') as f:
        f.write(info_str)

    data_module = DIVAHisDBDataModuleCropped(data_dir=datamodule_path)
    num_classes = data_module.num_classes
    class_encodings = data_module.class_encodings

    img_paths_per_page = CroppedHisDBDataset.get_gt_data_paths(directory=datamodule_path / 'test')

    dataset_img_name_list = []
    dataset_dict = defaultdict(list)
    for img_path, gt_path, img_name, pred_path, (x, y) in img_paths_per_page:
        if img_name not in dataset_img_name_list:
            dataset_img_name_list.append(img_name)
        dataset_dict[img_name].append((img_path, gt_path, pred_path, x, y))

    dataset_img_name_list = sorted(dataset_img_name_list)

    # sort dataset_dict lists
    for img_name in dataset_dict.keys():
        dataset_dict[img_name] = sorted(dataset_dict[img_name], key=lambda v: (v[4], v[3]))

    # Merge predictions on canvas
    img_name_list = sorted([str(n.name) for n in prediction_path.iterdir() if n.is_dir()])

    # check if all images from the dataset are found in the prediction output
    assert sorted(dataset_img_name_list) == sorted(img_name_list)

    pbar = tqdm(img_name_list)
    for img_name in pbar:
        pbar.set_description(f'Processing {img_name}')

        preds_folder = prediction_path / img_name
        coordinates = re.compile(r'.+_x(\d+)_y(\d+)\.npy$')

        if not preds_folder.is_dir():
            continue

        preds_list = []
        for pred_path in preds_folder.glob(f'{img_name}*.npy'):
            m = coordinates.match(pred_path.name)
            if m is None:
                continue
            x = int(m.group(1))
            y = int(m.group(2))
            preds_list.append((x, y, pred_path))
        preds_list = sorted(preds_list, key=lambda v: (v[1], v[0]))

        img_gt_list = dataset_dict[img_name]

        # The number of patches in the prediction should be equal to number of patches in dataset
        assert len(preds_list) == len(img_gt_list)

        crop_data_list = []
        # merge into one list
        for (x, y, pred_path), (img_path, gt_path, crop_name, x_data, y_data) in zip(preds_list, img_gt_list):
            assert (x, y) == (x_data, y_data)
            assert pred_path.name.startswith(crop_name)
            assert img_path.name.startswith(crop_name)
            assert gt_path.name.startswith(crop_name)

            pred = np.load(str(pred_path))
            crop_data_list.append(
                CropData(name=crop_name, offset_x=x, offset_y=y, width=pred.shape[1], height=pred.shape[2],
                         img_path=img_path, gt_path=gt_path, pred_path=pred_path))

        # Create new canvas
        canvas_width = crop_data_list[-1].width + crop_data_list[-1].offset_x
        canvas_height = crop_data_list[-1].height + crop_data_list[-1].offset_y

        pred_canvas_size = (num_classes, canvas_height, canvas_width)
        pred_canvas = np.empty(pred_canvas_size)
        pred_canvas.fill(np.nan)

        img_canvas = Image.new(mode='RGB', size=(canvas_width, canvas_height))
        gt_canvas = Image.new(mode='RGB', size=(canvas_width, canvas_height))

        for crop_data in tqdm(crop_data_list, desc='Merging crops', leave=False):
            # Add the pred to the pred_canvas
            pred = np.load(str(crop_data.pred_path))
            pred_canvas = merge_patches(pred, (crop_data.offset_x, crop_data.offset_y), pred_canvas)

            img_crop = pil_loader(crop_data.img_path)
            img_canvas.paste(img_crop, (crop_data.offset_x, crop_data.offset_y))

            gt_crop = pil_loader(crop_data.gt_path)
            gt_canvas.paste(gt_crop, (crop_data.offset_x, crop_data.offset_y))

        # Save the image when done
        outdir_img = output_path / 'img'
        outdir_gt = output_path / 'gt'
        outdir_pred = output_path / 'pred'

        outdir_img.mkdir(parents=True, exist_ok=True)
        outdir_gt.mkdir(parents=True, exist_ok=True)
        outdir_pred.mkdir(parents=True, exist_ok=True)

        outdir_gt_viz = output_path / 'gt_viz'
        outdir_gt_viz.mkdir(parents=True, exist_ok=True)
        outdir_pred_viz = output_path / 'pred_viz'
        outdir_pred_viz.mkdir(parents=True, exist_ok=True)

        img_canvas.save(fp=outdir_img / f'{img_name}.png')
        gt_canvas.save(fp=outdir_gt / f'{img_name}.png')
        visualize(img=str(outdir_gt / f'{img_name}.png'), out=str(outdir_gt_viz / f'{img_name}.png'))

        # Save prediction only when complete
        if not np.isnan(np.sum(pred_canvas)):
            # Save the final image (image_name, output_image, output_folder, class_encoding)
            save_output_page_image(image_name=f'{img_name}.png', output_image=pred_canvas,
                                   output_folder=outdir_pred, class_encoding=class_encodings)
            visualize(img=str(outdir_pred / f'{img_name}.png'), out=str(outdir_pred_viz / f'{img_name}.png'))
        else:
            print(f'WARNING: Test image {img_name} was not written! It still contains NaN values.')

    # Write final info
    info_str = f'- end_time:         \t{datetime.now():%Y-%m-%d_%H-%M-%S}'
    print('')
    print(info_str)
    print('')
    with info_file.open('a') as f:
        f.write(info_str)
        f.write('\n\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--datamodule_path',
                        help='Path to the root folder of the dataset (contains train/val/test)',
                        type=Path,
                        required=True)
    parser.add_argument('-p', '--prediction_path',
                        help='Path to the prediction patches folder',
                        type=Path,
                        required=True)
    parser.add_argument('-o', '--output_path',
                        help='Path to the output folder',
                        type=Path,
                        required=True)
    args = parser.parse_args()
    merge_cropped_output(**args.__dict__)
