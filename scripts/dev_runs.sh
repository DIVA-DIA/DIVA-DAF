#!/usr/bin/env bash
set -e

experiments="development_baby_unet_cb55_10
             development_baby_unet_rgb_data"

for exp in ${experiments}
do
  python run.py experiment="${exp}"
done
