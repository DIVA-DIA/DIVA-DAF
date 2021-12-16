#!/usr/bin/env bash
set -e

seeds="1544775463
       392423028"

params="experiment=rotnet_unet_cb55_full_pool"

for seed in ${seeds}
do
  python run.py ${params} +seed="${seed}"
done


seeds="3323771338
       1544775463
       392423028"

params="experiment=rotnet_unet_cb55_full_convpool"

for seed in ${seeds}
do
  python run.py ${params} +seed="${seed}"
done
