#!/usr/bin/env bash
set -e

seeds="914547167
       298402069
       656378804"

for seed in ${seeds}
do
  python run.py experiment=cb55_full_run_resnet50 +seed="${seed}"
  python run.py experiment=cb55_select_train15_run_resnet50 +seed="${seed}"
  python run.py experiment=cb55_select_train1_val1_run_resnet50 +seed="${seed}"
done
