from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

if __name__ == '__main__':
    gt_files_path_vh = Path("/net/research-hisdoc/datasets/self-supervised/CB55/binary_cleaned/sauvola/rlsa_vh_all_files")
    gt_files_path_rlsa = Path("/net/research-hisdoc/datasets/self-supervised/CB55/binary_cleaned/sauvola/savola_rlsa_25_median_5_all_files")
    output_path_vh = Path(
        "/net/research-hisdoc/datasets/self-supervised/CB55/binary_cleaned/sauvola/rlsa_vh_new_3cl_all_files")

    output_path_rlsa = Path(
        "/net/research-hisdoc/datasets/self-supervised/CB55/binary_cleaned/sauvola/rlsa_new_3cl_all_files")
    additional_space = 0.0
    threshold = 0.35

    space_from_side_border = 0.25
    channel_width = 0.25

    output_path_rlsa.mkdir(parents=True, exist_ok=True)
    output_path_vh.mkdir(parents=True, exist_ok=True)
    for gt_file_path_vh in tqdm(list(gt_files_path_vh.iterdir())):
        img_vh = Image.open(gt_file_path_vh)
        img_array_vh = np.logical_not(np.asarray(img_vh))
        img_rlsa = Image.open(gt_files_path_rlsa / gt_file_path_vh.name)
        img_array_rlsa = np.logical_not(np.asarray(img_rlsa))

        mid_col = int(img_vh.width / 2)
        mid_row = int(img_vh.height / 2)
        horizontal_pp = img_array_vh.sum(axis=0)
        vertical_pp = img_array_vh.sum(axis=1)

        while img_array_vh[mid_row, mid_col] == 0:
            mid_row = mid_row+10

        try:
            right = mid_col + np.argwhere(horizontal_pp[mid_col:] < (horizontal_pp[mid_col] * threshold))[0][0]
            # right = mid_col + np.argwhere(img_array[mid_row, mid_col:] == 0)[0][0]
            left_idx = np.argwhere(horizontal_pp[:mid_col] < (horizontal_pp[mid_col] * threshold))
            if not left_idx.any():
                left = 0
            else:
                left = np.argwhere(horizontal_pp[:mid_col] < (horizontal_pp[mid_col] * threshold))[-1][0]
            # left = np.argwhere(img_array[mid_row, :mid_col] == 0)[-1][0]
            top = np.argwhere(vertical_pp[:mid_row] < (vertical_pp[mid_row] * threshold))[-1][0]
            # top = np.argwhere(img_array[:mid_row, mid_col] == 0)[-1][0]
            bottom = mid_row + np.argwhere(vertical_pp[mid_row:] < (vertical_pp[mid_row] * threshold))[0][0]
            # bottom = mid_row + np.argwhere(img_array[mid_row:, mid_col] == 0)[0][0]
        except IndexError as e:
            print(f"index error with page{gt_file_path_vh}")
            print(e)
            continue
        offset_w = int((right - left) * additional_space)
        offset_h = int((bottom - top) * additional_space)

        img_array_new_vh = np.zeros((*img_array_vh.shape, 3)).astype(np.uint8)
        mask = np.zeros(img_array_vh.shape).astype(np.bool_)
        # img_array_new_vh[img_array_vh == 1] = (255, 0, 0)
        img_array_new_vh[img_array_vh == 1] = (0, 255, 255)
        mask[top - offset_h:bottom + offset_h, left - offset_w:right + offset_w] = img_array_vh[
                                                                                   top - offset_h:bottom + offset_h,
                                                                                   left - offset_w:right + offset_w] == 1
        # img_array_new_vh[mask] = (0, 255, 0)
        img_array_new_vh[mask] = (255, 255, 0)

        Image.fromarray(img_array_new_vh).save(output_path_vh / gt_file_path_vh.name)

        img_array_new_rlsa = np.zeros((*img_array_rlsa.shape, 3)).astype(np.uint8)
        mask = np.zeros(img_array_rlsa.shape).astype(np.bool_)
        # img_array_new_rlsa[img_array_rlsa == 1] = (0, 0, 255)
        img_array_new_rlsa[img_array_rlsa == 1] = (0, 255, 255)
        mask[top - offset_h:bottom + offset_h, left - offset_w:right + offset_w] = img_array_rlsa[
                                                                                   top - offset_h:bottom + offset_h,
                                                                                   left - offset_w:right + offset_w] == 1
        # img_array_new_rlsa[mask] = (255, 255, 0)
        img_array_new_rlsa[mask] = (255, 255, 0)
        Image.fromarray(img_array_new_rlsa).save(output_path_rlsa / gt_file_path_vh.name)
        # pass
