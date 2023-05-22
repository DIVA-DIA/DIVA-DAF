#!/usr/bin/env bash

set -e

epochs=("10" "20" "30" "40" "50")

for e in ${epochs[*]}; do
  for i in {0..9}; do
    params="experiment=binary_cb55_sauvola_rlsa_dataset1_unet.yaml
        trainer.devices=[0,1,2,3]
        trainer.max_epochs=${e}
        mode=hip.yaml
        name=PT_cb55_sauvola_rlsa_unet16_loss_no_weights_real_${e}epochs
        logger.wandb.project=hip
        logger.wandb.tags=[best_model,binary_rlsa,sauvola,rlsa,pre-training,unet16,dataset_rlsa_1,CB55-ssl-set,${e}-epochs]
        logger.wandb.group=pre-training-rlsa-${e}-ep"
    python run.py ${params}
  done
done
