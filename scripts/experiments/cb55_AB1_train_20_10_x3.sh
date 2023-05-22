#!/usr/bin/env bash

set -e

params_20="experiment=cb55_AB1_train_20_run_unet.yaml
        trainer.devices=[4,5,6,7]"
params_10="experiment=cb55_AB1_train_10_run_unet.yaml
        trainer.devices=[4,5,6,7]"


for i in {0..2}
do
  python run.py ${params_20}
done

for i in {0..2}
do
  python run.py ${params_10}
done
