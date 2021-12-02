#!/usr/bin/env bash
set -e

seeds="3323771338
       1544775463
       392423028"
params="experiment=development_baby_unet_cb55_10.yaml
        trainer.max_epochs=1"

for seed in ${seeds}
do
  python run.py ${params} seed="${seed}"
done
