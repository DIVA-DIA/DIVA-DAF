#!/usr/bin/env bash

set -e

bash scripts/experiments/ijdar/cb55_Ab1_training_5_adaptive_unet.sh
bash scripts/experiments/ijdar/cb55_Ab1_training_5_docufcn.sh
bash scripts/experiments/ijdar/cb55_Ab1_training_5_unet16.sh