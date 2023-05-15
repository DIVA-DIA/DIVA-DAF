#!/usr/bin/env bash

set -e

epochs=("10" "20" "30" "40" "50")

for e in ${epochs[*]}; do
  for i in {0..9}; do
    params="experiment=binary_cb55_sauvola_dataset1_unet.yaml
        trainer.devices=[2,3,5,6]
        trainer.max_epochs=${e}
        mode=hip.yaml
        name=PT_cb55_sauvola_unet16_loss_no_weights_${e}epochs
        logger.wandb.project=hip
        logger.wandb.tags=[best_model,binary,sauvola,pre-training,unet16,dataset1,CB55-ssl-set,${e}-epochs]
        logger.wandb.group=pre-sauvola-cb55-${e}-ep"
    python run.py ${params}
  done
done
