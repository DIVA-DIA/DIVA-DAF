#!/usr/bin/env bash

set -e

params_40="experiment=cb55_AB1_train_40_run_unet.yaml
        trainer.devices=[4,5,6,7]"

for i in {0..4}
do
  python run.py ${params_40}
done

