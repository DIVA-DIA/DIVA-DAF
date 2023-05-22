#!/usr/bin/env bash

set -e

epochs=("10" "20" "30" "40" "50")

for e in ${epochs[*]}; do
  for i in {0..9}; do
    params="experiment=binary_cb55_gaussian_dataset1_unet.yaml
        trainer.devices=[0,1,2,3]
        trainer.max_epochs=${e}
        name=binary_cb55_gaussian_unet_loss_no_weights_${e}epochs
        logger.wandb.tags=[best_model,binary,gaussian,pre-training,unet,dataset1,CB55-ssl-set,${e}-epochs]
        logger.wandb.group=pre-training-${e}-ep"
    python run.py ${params}
  done
done
