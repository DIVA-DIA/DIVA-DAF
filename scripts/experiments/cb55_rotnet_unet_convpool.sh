#!/usr/bin/env bash
set -e

seeds="3323771338
       1544775463
       392423028"

params="experiment=rotnet_unet_cb55_full_convpool trainer.gpus=[0,1,2,3] datamodule.batch_size=100"

for seed in ${seeds}
do
  python run.py ${params} +seed="${seed}"
done
