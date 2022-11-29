#!/usr/bin/env bash

set -e

params_40="experiment=cb55_AB1_train_40_run_divanet.yaml
        trainer.devices=[4,6,7,8]"
params_20="experiment=cb55_AB1_train_20_run_divanet.yaml
        trainer.devices=[4,6,7,8]"
params_10="experiment=cb55_AB1_train_10_run_divanet.yaml
        trainer.devices=[4,6,7,8]"

for i in {0..4}
do
  python run.py ${params_40}
done

for i in {0..4}
do
  python run.py ${params_20}
done

for i in {0..4}
do
  python run.py ${params_10}
done
