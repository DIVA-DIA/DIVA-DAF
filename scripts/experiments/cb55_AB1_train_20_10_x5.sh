#!/usr/bin/env bash

set -e

params_20="experiment=cb55_AB1_train_20_run_unet.yaml
        trainer.devices=[0,1,2,3]"
params_10="experiment=cb55_AB1_train_10_run_unet.yaml
        trainer.devices=[0,1,2,3]"


for i in {0..4}
do
  python run.py ${params_20}
done

for i in {0..4}
do
  python run.py ${params_10}
done
