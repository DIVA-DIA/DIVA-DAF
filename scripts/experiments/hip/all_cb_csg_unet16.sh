#!/usr/bin/env bash

set -e

bash scripts/experiments/hip/cb55_sauvola_rlsa_pretrain_x10.sh
bash scripts/experiments/hip/csg18_sauvola_pretrain_x10.sh
bash scripts/experiments/hip/csg18_sauvola_rlsa_pretrain_x10.sh
bash scripts/experiments/hip/csg863_sauvola_pretrain_x10.sh
bash scripts/experiments/hip/csg863_sauvola_rlsa_pretrain_x10.sh
bash scripts/experiments/hip/heuristic_sauvola_rlsa_pretrain_x10.sh