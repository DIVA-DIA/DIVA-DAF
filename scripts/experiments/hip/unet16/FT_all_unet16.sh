#!/usr/bin/env bash

set -e


bash scripts/experiments/hip/unet16/fine_tune_sauvola_cb55_polygon_unet16_10_20_30_40_50.sh
bash scripts/experiments/hip/unet16/fine_tune_sauvola_csg18_polygon_unet16_10_20_30_40_50.sh
bash scripts/experiments/hip/unet16/fine_tune_sauvola_csg863_polygon_unet16_10_20_30_40_50.sh
bash scripts/experiments/hip/unet16/fine_tune_sauvola_heuristic_cb55_polygon_unet16_10_20_30_40_50.sh
bash scripts/experiments/hip/unet16/fine_tune_sauvola_heuristic_csg18_863_polygon_unet16_10_20_30_40_50.sh
bash scripts/experiments/hip/unet16/fine_tune_sauvola_rlsa_cb55_polygon_unet16_10_30_40_50.sh
bash scripts/experiments/hip/unet16/fine_tune_sauvola_rlsa_csg18_polygon_unet16_10_20_30_40_50.sh
bash scripts/experiments/hip/unet16/fine_tune_sauvola_rlsa_csg863_polygon_unet16_10_20_30_40_50.sh