import argparse
import math
import re
import threading
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from multiprocessing.pool import ThreadPool
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from src.datamodules.DivaHisDB.datamodule_cropped import DivaHisDBDataModuleCropped
from src.datamodules.DivaHisDB.datasets.cropped_dataset import CroppedHisDBDataset
from src.datamodules.DivaHisDB.utils.output_tools import save_output_page_image
from src.datamodules.utils.output_tools import merge_patches
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


class CroppedOutputMerger:
    def __init__(self, datamodule_path: Path, prediction_path: Path, output_path: Path,
                 data_folder_name: str, gt_folder_name: str, num_threads: int = 10):
        # Defaults
        self.load_only_first_crop_for_size = True  # All crops have to be the same size in the current implementation

        self.datamodule_path = datamodule_path
        self.prediction_path = prediction_path
        self.output_path = output_path

        self.data_folder_name = data_folder_name
        self.gt_folder_name = gt_folder_name

        data_module = DivaHisDBDataModuleCropped(data_dir=str(datamodule_path), data_folder_name=self.data_folder_name,
                                                 gt_folder_name=self.gt_folder_name)
        self.num_classes = data_module.num_classes
        self.class_encodings = data_module.class_encodings

        img_paths_per_page = CroppedHisDBDataset.get_gt_data_paths(directory=datamodule_path / 'test',
                                                                   data_folder_name=self.data_folder_name,
                                                                   gt_folder_name=self.gt_folder_name)

        coordinates = re.compile(r'.+_x(\d+)_y(\d+)$')

        dataset_img_name_list = []
        self.dataset_dict = defaultdict(list)
        for img_path, gt_path, img_name, crop_name in img_paths_per_page:
            if img_name not in dataset_img_name_list:
                dataset_img_name_list.append(img_name)
            m = coordinates.match(crop_name)
            if m is None:
                continue
            x = int(m.group(1))
            y = int(m.group(2))
            self.dataset_dict[img_name].append((img_path, gt_path, crop_name, x, y))

        dataset_img_name_list = sorted(dataset_img_name_list)

        # sort dataset_dict lists
        for img_name in self.dataset_dict.keys():
            self.dataset_dict[img_name] = sorted(self.dataset_dict[img_name], key=lambda v: (v[4], v[3]))

        self.img_name_list = sorted([str(n.name) for n in prediction_path.iterdir() if n.is_dir()])

        # check if all images from the dataset are found in the prediction output
        assert sorted(dataset_img_name_list) == sorted(self.img_name_list)

        self.num_pages = len(self.img_name_list)
        if self.num_pages >= num_threads:
            self.num_threads = num_threads
        else:
            self.num_threads = self.num_pages

        assert self.num_pages > 0

    def merge_all(self):
        start_time = datetime.now()
        info_list = ['Running merge_cropped_output_HisDB.py:',
                     f'- start_time:                    \t{start_time:%Y-%m-%d_%H-%M-%S}',
                     f'- datamodule_path:               \t{self.datamodule_path}',
                     f'- prediction_path:               \t{self.prediction_path}',
                     f'- output_path:                   \t{self.output_path}',
                     f'- num_pages:                     \t{self.num_pages}',
                     f'- num_threads:                   \t{self.num_threads}',
                     '']  # empty string to get linebreak at the end when using join
        info_str = '\n'.join(info_list)
        print(info_str, flush=True)

        # Write info_cropped_dataset.txt
        self.output_path.mkdir(parents=True, exist_ok=True)
        info_file = self.output_path / 'info_merge_cropped_output.txt'
        with info_file.open('a') as f:
            f.write(info_str)

        pool = ThreadPool(self.num_threads)
        lock = threading.Lock()
        results = []
        for position, img_name in enumerate(self.img_name_list):
            results.append(pool.apply_async(self.merge_page, args=(img_name, lock, position)))
        pool.close()
        pool.join()

        results = [r.get() for r in results]

        # Closing the progress bars in order for a beautiful output
        for i in range(3):
            for pbars in results:
                pbars[i].close()

        end_time = datetime.now()
        duration = end_time - start_time

        # Write final info
        info_list = [f'- end_time:                      \t{datetime.now():%Y-%m-%d_%H-%M-%S}',
                     f'- duration:                      \t{duration}',
                     '']  # empty string to get linebreak at the end when using join
        info_str = '\n'.join(info_list)

        print('\n' + info_str)
        # print(f'- log_file:                      \t{info_file}\n')

        with info_file.open('a') as f:
            f.write(info_str)
            f.write('\n')

        print('Evaluation script command:')
        print(f'python tools/evaluate_algorithm.py'
              f' --gt_folder {self.output_path / "gt"}'
              f' --prediction_folder {self.output_path / "pred"}'
              f' --original_images {self.output_path / "img"}'
              f' --output_path analysis'
              f'\n')

        print('DONE!')

    def merge_page(self, img_name: str, lock, position):
        page_info_str = f'[{str(position + 1).rjust(int(math.log10(self.num_pages)) + 1)}/{self.num_pages}] {img_name}'

        preds_folder = self.prediction_path / img_name
        coordinates = re.compile(r'.+_x(\d+)_y(\d+)\.npy$')

        if not preds_folder.is_dir():
            print(f'Skipping {preds_folder}. Not a directory!')
            return

        preds_list = []
        for pred_path in preds_folder.glob(f'{img_name}*.npy'):
            m = coordinates.match(pred_path.name)
            if m is None:
                continue
            x = int(m.group(1))
            y = int(m.group(2))
            preds_list.append((x, y, pred_path))
        preds_list = sorted(preds_list, key=lambda v: (v[1], v[0]))

        img_gt_list = self.dataset_dict[img_name]

        # The number of patches in the prediction should be equal to number of patches in dataset
        assert len(preds_list) == len(img_gt_list)

        crop_data_list = []

        # merge into one list
        with lock:
            pbar1 = tqdm(total=len(preds_list),
                         position=position,
                         # file=sys.stdout,
                         leave=True,
                         desc=f'{page_info_str}: Merging path lists')

        crop_width = -1
        crop_height = -1

        if self.load_only_first_crop_for_size:
            pred_path = preds_list[0][2]
            pred = np.load(str(pred_path))
            crop_width = pred.shape[1]
            crop_height = pred.shape[2]

        for (x, y, pred_path), (img_path, gt_path, crop_name, x_data, y_data) in zip(preds_list, img_gt_list):
            assert (x, y) == (x_data, y_data)
            assert pred_path.name.startswith(crop_name)
            assert img_path.name.startswith(crop_name)
            assert gt_path.name.startswith(crop_name)

            if not self.load_only_first_crop_for_size:
                pred = np.load(str(pred_path))
                crop_width = pred.shape[1]
                crop_height = pred.shape[2]

            crop_data_list.append(
                CropData(name=crop_name, offset_x=x, offset_y=y, width=crop_width, height=crop_height,
                         img_path=img_path, gt_path=gt_path, pred_path=pred_path))  # , pred=pred))

            pbar1.update()

        with lock:
            pbar1.refresh()

        # Create new canvas
        canvas_width = crop_data_list[-1].width + crop_data_list[-1].offset_x
        canvas_height = crop_data_list[-1].height + crop_data_list[-1].offset_y

        pred_canvas_size = (self.num_classes, canvas_height, canvas_width)
        pred_canvas = np.empty(pred_canvas_size)
        pred_canvas.fill(np.nan)

        img_canvas = Image.new(mode='RGB', size=(canvas_width, canvas_height))
        gt_canvas = Image.new(mode='RGB', size=(canvas_width, canvas_height))

        with lock:
            pbar2 = tqdm(total=len(crop_data_list),
                         position=position + (1 * self.num_pages),
                         # file=sys.stdout,
                         leave=True,
                         desc=f'{page_info_str}: Merging crops')

        for crop_data in crop_data_list:
            # Add the pred to the pred_canvas
            pred = np.load(str(crop_data.pred_path))

            # make sure all crops have same size
            assert crop_width == pred.shape[1]
            assert crop_height == pred.shape[2]

            pred_canvas = merge_patches(pred, (crop_data.offset_x, crop_data.offset_y), pred_canvas)

            img_crop = pil_loader(crop_data.img_path)
            img_canvas.paste(img_crop, (crop_data.offset_x, crop_data.offset_y))

            gt_crop = pil_loader(crop_data.gt_path)
            gt_canvas.paste(gt_crop, (crop_data.offset_x, crop_data.offset_y))

            pbar2.update()

        with lock:
            pbar2.refresh()

        # Save the image when done
        outdir_img = self.output_path / 'img'
        outdir_gt = self.output_path / 'gt'
        outdir_pred = self.output_path / 'pred'

        outdir_img.mkdir(parents=True, exist_ok=True)
        outdir_gt.mkdir(parents=True, exist_ok=True)
        outdir_pred.mkdir(parents=True, exist_ok=True)

        outdir_gt_viz = self.output_path / 'gt_viz'
        outdir_gt_viz.mkdir(parents=True, exist_ok=True)
        outdir_pred_viz = self.output_path / 'pred_viz'
        outdir_pred_viz.mkdir(parents=True, exist_ok=True)

        # Loop to allow progress bar
        with lock:
            pbar3 = tqdm(total=5,
                         position=position + (2 * self.num_pages),
                         # file=sys.stdout,
                         leave=True,
                         desc=f'{page_info_str}: Saving merged image files')

        for i in range(5):
            if i == 0:
                pbar3.set_description(f'{page_info_str}: Saving merged image files ' + '(img)'.ljust(10))
                img_canvas.save(fp=outdir_img / f'{img_name}.png')

            elif i == 1:
                pbar3.set_description(f'{page_info_str}: Saving merged image files ' + '(gt)'.ljust(10))
                gt_canvas.save(fp=outdir_gt / f'{img_name}.png')

            elif i == 2:
                pbar3.set_description(f'{page_info_str}: Saving merged image files ' + '(gt_viz)'.ljust(10))
                visualize(img=str(outdir_gt / f'{img_name}.png'), out=str(outdir_gt_viz / f'{img_name}.png'))

            elif i == 3:
                pbar3.set_description(f'{page_info_str}: Saving merged image files ' + '(pred)'.ljust(10))
                # Save prediction only when complete
                if not np.isnan(np.sum(pred_canvas)):
                    # Save the final image (image_name, output_image, output_folder, class_encoding)
                    save_output_page_image(image_name=f'{img_name}.png', output_image=pred_canvas,
                                           output_folder=outdir_pred, class_encoding=self.class_encodings)
                else:
                    print(f'WARNING: Test image {img_name} was not written! It still contains NaN values.')
                    break  # so last step is not

            elif i == 4:
                pbar3.set_description(f'{page_info_str}: Saving merged image files ' + '(pred_viz)'.ljust(10))
                if (outdir_pred / f'{img_name}.png').exists():
                    visualize(img=str(outdir_pred / f'{img_name}.png'), out=str(outdir_pred_viz / f'{img_name}.png'))

            pbar3.update()

        with lock:
            pbar3.refresh()

        # The progress bars will be close in order in main thread
        return pbar1, pbar2, pbar3


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
    parser.add_argument('-df', '--data_folder_name',
                        help='Name of data folder',
                        type=str,
                        required=True)
    parser.add_argument('-gf', '--gt_folder_name',
                        help='Name of gt folder',
                        type=str,
                        required=True)
    parser.add_argument('-n', '--num_threads',
                        help='Number of threads for parallel processing',
                        type=int,
                        default=10)

    args = parser.parse_args()
    merger = CroppedOutputMerger(**args.__dict__)
    merger.merge_all()
