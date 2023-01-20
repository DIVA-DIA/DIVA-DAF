#!/usr/bin/env bash

set -e

epochs=("200")

for e in ${epochs[*]}; do
  for i in {0..9}; do
    params="experiment=morpho_cb55_B22_unet.yaml
        trainer.devices=[0,1,2,3]
        trainer.max_epochs=${e}
        name=morpho_cb55_B22_unet_loss_no_weights_${e}epochs
        logger.wandb.tags=[best_model,morpho,B22,pre-training,unet,B22_dataset_2_icdar,CB55-ssl-set,${e}-epochs]
        logger.wandb.group=pre-training-morpho-b22-${e}-ep"
    python run.py ${params}
  done
done
