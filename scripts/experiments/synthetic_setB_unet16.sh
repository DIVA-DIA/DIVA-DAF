#!/usr/bin/env bash

set -e

params_unet16="experiment=synthetic_DPC_unet16.yaml
          trainer.devices=[4,5,6,7]
          logger.wandb.tags=[unet16,4-classes,pre-training,200-epochs,no-weights,no-init,DPC60-SetB]"

python run.py ${params_unet16}
