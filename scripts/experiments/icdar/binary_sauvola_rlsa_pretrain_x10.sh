#!/usr/bin/env bash

set -e

epochs=("10" "20" "30" "40" "50")

for e in ${epochs[*]}; do
  for i in {0..9}; do
    params="experiment=binary_cb55_sauvola_rlsa_dataset1_unet.yaml
        trainer.devices=[4,5,6,7]
        trainer.max_epochs=${e}
        name=binary_cb55_sauvola_rlsa_unet_loss_no_weights_real_${e}epochs
        logger.wandb.tags=[best_model,binary_rlsa,sauvola,rlsa,pre-training,unet,dataset_rlsa_1,CB55-ssl-set,${e}-epochs]
        logger.wandb.group=pre-training-rlsa-${e}-ep"
    python run.py ${params}
  done
done
