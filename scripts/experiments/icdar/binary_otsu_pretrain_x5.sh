#!/usr/bin/env bash

set -e

epochs=("30" "40")

for e in ${epochs[*]}; do
  for i in {0..9}; do
    params="experiment=binary_cb55_otsu_dataset1_unet.yaml
        trainer.devices=[4,5,6,7]
        trainer.max_epochs=${e}
        name=binary_cb55_otsu_unet_loss_no_weights_${e}epochs
        logger.wandb.tags=[best_model,binary,otsu,pre-training,unet,dataset1,CB55-ssl-set,${e}-epochs]
        logger.wandb.group=pre-training-otsu-${e}-ep"
    python run.py ${params}
  done
done
