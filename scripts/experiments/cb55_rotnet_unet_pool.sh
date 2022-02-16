#!/usr/bin/env bash
set -e

seeds="1544775463
       392423028"

params="experiment=rotnet_unet_cb55_full_pool trainer.gpus=[0,1,2,3]"

for seed in ${seeds}
do
  python run.py ${params} +seed="${seed}"
done
