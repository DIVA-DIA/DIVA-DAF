#!/usr/bin/env bash

set -e

bash scripts/experiments/cb55_Ab1_training_10_unet.sh
bash scripts/experiments/cb55_Ab1_training_10_unet16.sh
bash scripts/experiments/cb55_Ab1_training_10_unet32.sh
bash scripts/experiments/cb55_Ab1_training_10_unet64.sh
