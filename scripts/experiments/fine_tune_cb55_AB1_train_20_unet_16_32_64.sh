#!/usr/bin/env bash

set -e

params_unet="experiment=fine_tune_cb55_AB1_train_20_run_unet.yaml
        trainer.devices=[4,5,6,7]"


for i in {0..0}
do
  python run.py ${params_unet}
done

params_unet16="experiment=fine_tune_cb55_AB1_train_20_run_unet16.yaml
        trainer.devices=[4,5,6,7]"


for i in {0..0}
do
  python run.py ${params_unet16}
done

params_unet32="experiment=fine_tune_cb55_AB1_train_20_run_unet32.yaml
        trainer.devices=[4,5,6,7]"


for i in {0..0}
do
  python run.py ${params_unet32}
done

params_unet64="experiment=fine_tune_cb55_AB1_train_20_run_unet64.yaml
        trainer.devices=[4,5,6,7]"


for i in {0..0}
do
  python run.py ${params_unet64}
done

params_divanet="experiment=fine_tune_cb55_AB1_train_20_run_divanet.yaml
        trainer.devices=[4,5,6,7]"


for i in {0..0}
do
  python run.py ${params_divanet}
done