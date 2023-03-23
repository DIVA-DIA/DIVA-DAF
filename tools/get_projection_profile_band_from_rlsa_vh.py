import itertools
from pathlib import Path

import numpy as np
from typing import Dict
from PIL import Image, ImageDraw
from skimage.measure import regionprops, label
from multiprocessing import Pool, cpu_count

MAIN_TEXT_COLOR = (255, 255, 0)
GLOSS_COLOR = (0, 255, 255)


def adapt_cc(binary_img_vh, mask_input, left_pos, right_pos, r_band_r, l_band_l, top_pos, bottom_pos,
             t_band_t, b_band_b):
    threshold_v = 0.05
    threshold_h = 0.15

    labelled_main = label(binary_img_vh[top_pos:bottom_pos, left_pos:right_pos].astype(np.uint8))
    props_main = regionprops(labelled_main)

    labelled_right = label(binary_img_vh[:, right_pos:].astype(np.uint8))
    props_right = regionprops(labelled_right)

    labelled_left = label(binary_img_vh[:, :left_pos].astype(np.uint8))
    props_left = regionprops(labelled_left)

    labelled_top = label(binary_img_vh[:top_pos, left_pos:right_pos].astype(np.uint8))
    props_top = regionprops(labelled_top)

    labelled_bottom = label(binary_img_vh[bottom_pos:, left_pos:right_pos].astype(np.uint8))
    props_bottom = regionprops(labelled_bottom)

    main_area = sum([p.area for p in props_main])

    for p in props_right:
        if p.bbox[1] != 0 or p.area > main_area * threshold_v or p.bbox[3] + right_pos > r_band_r:
            continue
        # get all trues in the bbox
        mask = np.zeros(binary_img_vh.shape).astype(np.bool_)
        mask[p.bbox[0]: p.bbox[2], right_pos + p.bbox[1]: right_pos + p.bbox[3]] = binary_img_vh[p.bbox[0]:p.bbox[2],
                                                                                   right_pos + p.bbox[1]: right_pos +
                                                                                                          p.bbox[
                                                                                                              3]] == 1
        mask_input[mask] = 1

    for p in props_left:
        if p.bbox[3] != left_pos or p.area > main_area * threshold_v or p.bbox[1] < l_band_l:
            continue
        # get all trues in the bbox
        mask = np.zeros(binary_img_vh.shape).astype(np.bool_)
        mask[p.bbox[0]: p.bbox[2], p.bbox[1]: p.bbox[3]] = binary_img_vh[p.bbox[0]:p.bbox[2], p.bbox[1]:p.bbox[3]] == 1
        mask_input[mask] = 1

    for p in props_top:
        if p.bbox[2] != top_pos or p.area > main_area * threshold_h or p.bbox[2] < t_band_t:
            continue
        # get all trues in the bbox
        mask = np.zeros(binary_img_vh.shape).astype(np.bool_)
        mask[p.bbox[0]: p.bbox[2],
        left_pos + p.bbox[1]: left_pos + p.bbox[3]] = binary_img_vh[p.bbox[0]:p.bbox[2],
                                                      left_pos + p.bbox[1]:left_pos + p.bbox[3]] == 1
        mask_input[mask] = 1

    for p in props_bottom:
        if p.bbox[0] != 0 or p.area > main_area * threshold_h or p.bbox[2] > b_band_b:
            continue
        # get all trues in the bbox
        mask = np.zeros(binary_img_vh.shape).astype(np.bool_)
        mask[bottom_pos + p.bbox[0]:bottom_pos + p.bbox[2],
        left_pos + p.bbox[1]:left_pos + p.bbox[3]] = binary_img_vh[bottom_pos + p.bbox[0]:bottom_pos + p.bbox[2],
                                                     left_pos + p.bbox[1]:left_pos + p.bbox[3]] == 1
        mask_input[mask] = 1


def get_band_dimensions(pixel_from_border: float, pixel_band: float):
    return (int(pixel_from_border - pixel_band),
            int(pixel_from_border + pixel_band))


def pp_classification(file_path: Path, output_rlsa: Path, output_vh: Path, spacing: Dict[str, Dict[str, int]],
                      bands: bool):
    print("pp for file: ", file_path)
    img_vh = Image.open(file_path)
    img_array_vh = np.logical_not(np.asarray(img_vh))

    img_rlsa = Image.open(gt_files_path_rlsa / file_path.name)
    img_array_rlsa = np.logical_not(np.asarray(img_rlsa))

    horizontal_pp = img_array_vh.sum(axis=0)

    l_band_l, l_band_r = get_band_dimensions(pixel_from_border=spacing['left']['line'],
                                             pixel_band=spacing['left']['width'])
    r_band_l, r_band_r = get_band_dimensions(pixel_from_border=spacing['right']['line'],
                                             pixel_band=spacing['right']['width'])
    t_band_t, t_band_b = get_band_dimensions(pixel_from_border=spacing['top']['line'],
                                             pixel_band=spacing['top']['width'])
    b_band_t, b_band_b = get_band_dimensions(pixel_from_border=spacing['bottom']['line'],
                                             pixel_band=spacing['bottom']['width'])

    l_band_l = l_band_l
    l_band_r = l_band_r
    left_band = horizontal_pp[l_band_l:l_band_r]
    right_band = horizontal_pp[r_band_l:r_band_r]

    left_idx = get_left_index(left_band)
    # right_max_idx, right_min_idx = get_right_index(right_band)
    right_idx = get_right_index(right_band)

    img_array_new_vh = np.zeros((*img_array_vh.shape, 3)).astype(np.uint8)
    mask_h = np.zeros(img_array_vh.shape).astype(np.bool_)
    # img_array_new_vh[img_array_vh == 1] = (255, 0, 0)
    img_array_new_vh[img_array_vh == 1] = GLOSS_COLOR
    mask_h[:, l_band_l + left_idx:r_band_l + right_idx] = img_array_vh[:, l_band_l + left_idx:r_band_l + right_idx] == 1

    mask_rlsa = mask_h.copy()
    mask_rlsa[:, l_band_l + left_idx:r_band_l + right_idx] = img_array_vh[:,
                                                             l_band_l + left_idx:r_band_l + right_idx] == 1
    vertical_pp = mask_rlsa.sum(axis=1)
    top_band = vertical_pp[t_band_t:t_band_b]
    bottom_band = vertical_pp[b_band_t:b_band_b]
    top_idx = get_top_index(top_band)
    bottom_idx = get_bottom_index(bottom_band)

    img_array_vh_ad = img_array_vh.copy()
    img_array_vh_ad[np.logical_not(mask_h)] = 0
    main_text_mask = np.zeros(img_array_vh_ad.shape).astype(np.bool_)
    main_text_mask[t_band_t + top_idx: b_band_t + bottom_idx, :] = img_array_vh_ad[
                                                                  t_band_t + top_idx: b_band_t + bottom_idx, :] == 1

    adapt_cc(img_array_rlsa, mask_input=main_text_mask,
             left_pos=l_band_l + left_idx, right_pos=r_band_l + right_idx, r_band_r=r_band_r, l_band_l=l_band_l,
             top_pos=t_band_t + top_idx, bottom_pos=b_band_t + bottom_idx, t_band_t=t_band_t, b_band_b=b_band_b)
    remove_noise(main_text_mask)
    comments_mask = img_array_vh.copy()
    comments_mask[main_text_mask] = 0
    remove_noise(comments_mask)

    image_vh = np.zeros((*img_array_vh.shape, 3)).astype(np.uint8)
    image_vh[main_text_mask] = MAIN_TEXT_COLOR
    image_vh[comments_mask] = GLOSS_COLOR
    # adapt_cc(img_array_vh, right_band_left, right_max_idx, right_min_idx)
    img_vh = Image.fromarray(image_vh)
    if bands:
        draw_bands(img_vh, spacing)
    img_vh.save(output_vh / file_path.name)

    remove_noise(img_array_rlsa)
    image_vh[np.logical_not(img_array_rlsa)] = [0, 0, 0]
    img_rlsa = Image.fromarray(image_vh)
    if bands:
        draw_bands(img_rlsa, spacing)
    img_rlsa.save(output_rlsa / file_path.name)


def remove_noise(img_array: np.ndarray):
    threshold = 1000
    labelled = label(img_array)
    props = regionprops(labelled)

    for p in props:
        if p.area < threshold:
            img_array[p.bbox[0]:p.bbox[2],
                      p.bbox[1]:p.bbox[3]] = 0


def draw_bands(img_vh, spacing):
    draw_vh = ImageDraw.Draw(img_vh)

    draw_vh.line((spacing['left']['line'], 0, spacing['left']['line'], img_vh.height), fill=(255, 0, 0), width=5)

    draw_vh.line((spacing['right']['line'], 0, spacing['right']['line'], img_vh.height), fill=(255, 0, 0), width=5)

    draw_vh.line((0, spacing['top']['line'], img_vh.width, spacing['top']['line']), fill=(255, 0, 0), width=5)

    draw_vh.line((0, spacing['bottom']['line'], img_vh.width, spacing['bottom']['line']), fill=(255, 0, 0), width=5)


def get_top_index(top_band):
    new_top_band = np.roll(top_band, -1)
    difference = new_top_band[:-1] - top_band[:-1]
    return np.argmax(new_top_band[:-1] - top_band[:-1])


def get_bottom_index(bottom_band):
    new_bottom_band = np.roll(bottom_band, 1)

    return np.argmax(new_bottom_band[1:] - bottom_band[1:])


def get_right_index(right_band):
    new_right_band = np.roll(right_band, 1)
    # difference = new_right_band[1:] - right_band[1:]
    # max_dif_idx = np.argmax(difference)
    # static number not working as some pages are tilted (c)
    return np.argmin(new_right_band)  # max_dif_idx  # , np.argmin(difference[max_dif_idx:])


def get_left_index(left_band):
    new_left_band = np.roll(left_band, -1)

    return np.argmax(new_left_band[:-1] - left_band[:-1])


if __name__ == '__main__':
    gt_files_path_vh = Path(
        "/net/research-hisdoc/datasets/self-supervised/CB55/binary_cleaned/sauvola/rlsa_vh_all_files")
    gt_files_path_rlsa = Path(
        "/net/research-hisdoc/datasets/self-supervised/CB55/binary_cleaned/sauvola/savola_rlsa_25_median_5_all_files")
    output_path_vh = Path(
        "/net/research-hisdoc/datasets/self-supervised/CB55/binary_cleaned/sauvola/rlsa_vh_new_3cl_all_files")

    output_path_rlsa = Path(
        "/net/research-hisdoc/datasets/self-supervised/CB55/binary_cleaned/sauvola/rlsa_new_3cl_all_files")
    additional_space = 0.0
    threshold_pp = 0.35
    drawing_red_lines = False

    stdev_times = 2

    spacing = {
        "top": {"line": 165, "width": stdev_times * 6},
        "bottom": {"line": 1344 - 332, "width": stdev_times * 22},
        "left": {"line": 241, "width": stdev_times * 13},
        "right": {"line": 960 - 240, "width": stdev_times * 14}
    }

    # space_from_side_border_w = 0.25
    # band_width = 0.10
    #
    # space_from_side_border_h = 0.15
    # band_height = 0.075

    output_path_rlsa.mkdir(parents=True, exist_ok=True)
    output_path_vh.mkdir(parents=True, exist_ok=True)

    p = Pool(cpu_count())
    # [25:26]
    # 64r: [28:29]
    p.starmap(pp_classification, zip(sorted(list(gt_files_path_vh.iterdir())),
                                     itertools.repeat(output_path_rlsa),
                                     itertools.repeat(output_path_vh),
                                     itertools.repeat(spacing),
                                     itertools.repeat(drawing_red_lines)))
