#!/usr/bin/env bash

set -e

bash scripts/experiments/hip/fine_tune_sauvola_cb55_polygon_10_20_30_40_50.sh
bash scripts/experiments/hip/fine_tune_sauvola_rlsa_cb55_polygon_10_30_40_50.sh
bash scripts/experiments/hip/fine_tune_sauvola_rlsa_new_3cl_cb55_polygon_10_20_30_40_50.sh
