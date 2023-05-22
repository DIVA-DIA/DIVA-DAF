#!/usr/bin/env bash

set -e

epochs=("10" "20" "30" "40" "50")

#for e in ${epochs[*]}; do
for i in {0..9}; do
  params="experiment=binary_cb55_sauvola_cleaned_dataset1_unet.yaml
        trainer.devices=[4,5,6,7]"
  python run.py ${params}
done
#done
