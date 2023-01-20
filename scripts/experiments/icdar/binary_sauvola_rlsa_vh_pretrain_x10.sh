#!/usr/bin/env bash

set -e

epochs=("50")

for e in ${epochs[*]}; do
  for i in {0..9}; do
    params="experiment=binary_cb55_sauvola_rlsa_vh_dataset1_unet.yaml
        trainer.devices=[4,5,6,7]
        trainer.max_epochs=${e}
        name=binary_cb55_sauvola_rlsa_vh_unet_loss_no_weights_${e}epochs
        logger.wandb.tags=[best_model,binary_rlsa_vh,sauvola,rlsa_vh,pre-training,unet,dataset_rlsa_vh_1,CB55-ssl-set,${e}-epochs]
        logger.wandb.group=pre-training-rlsa-vh-${e}-ep"
    python run.py ${params}
  done
done
