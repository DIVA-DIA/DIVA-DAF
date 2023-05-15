#!/usr/bin/env bash

set -e

epochs=("10" "20" "30" "40" "50")

for e in ${epochs[*]}; do
  for i in {0..9}; do
    params="experiment=binary_csg863_sauvola_rlsa_dataset_1_960_1440_unet.yaml
        trainer.devices=[0,1,2,3]
        trainer.max_epochs=${e}
        mode=hip.yaml
        name=PT_csg863_sauvola_rlsa_unet16_loss_no_weights_${e}epoch
        logger.wandb.project=hip
        logger.wandb.tags=[best_model,rlsa,sauvola,pre-training,unet16,dataset_rlsa_1,CSG863-ssl-set,${e}-epochs]
        logger.wandb.group=pre-training-rlsa-3cl-csg863-${e}-ep"
    python run.py ${params}
  done
done
