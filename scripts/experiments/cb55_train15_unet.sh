#!/usr/bin/env bash
set -e

seeds="3323771338
       1544775463
       392423028"
params="experiment=cb55_select_train15_unet
        trainer.check_val_every_n_epoch=4
        trainer.max_epochs=50"

for seed in ${seeds}
do
  python run.py ${params} +seed="${seed}"
done
