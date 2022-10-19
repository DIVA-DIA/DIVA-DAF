#!/usr/bin/env bash
set -e

params="experiment=morpho_cb55_A11_unet.yaml"


for i in {1..4}
do
  python run.py ${params}
done
