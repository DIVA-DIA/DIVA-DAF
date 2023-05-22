#!/usr/bin/env bash

set -e

params_40="experiment=cb55_AB1_train_40_run_unet.yaml trainer.devices=[0,1,2,3]"

for i in {0..2}
do
  python run.py ${params_40}
done

