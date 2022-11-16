#!/usr/bin/env bash

set -e

params_dummy="experiment=synthetic_dummy_unet.yaml trainer.devices=[0,1,2,3]"
params_mixed="experiment=synthetic_mixed_synth_unet.yaml trainer.devices=[0,1,2,3]"
params_mixed_real="experiment=synthetic_mixed_real_unet.yaml trainer.devices=[0,1,2,3]"

for i in {0..4}
do
  python run.py ${params_dummy}
done

for i in {0..4}
do
  python run.py ${params_mixed}
done

for i in {0..4}
do
  python run.py ${params_mixed_real}
done
