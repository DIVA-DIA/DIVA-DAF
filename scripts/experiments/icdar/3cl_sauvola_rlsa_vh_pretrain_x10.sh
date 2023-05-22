#!/usr/bin/env bash

set -e

epochs=("10" "20" "30" "40")

for e in ${epochs[*]}; do
  for i in {0..9}; do
    params="experiment=cb55_sauvola_rlsa_vh_3cl_dataset1_unet.yaml
        trainer.devices=[0,1,2,3]
        trainer.max_epochs=${e}
        name=3cl_rlsa_vh_cb55_sauvola_unet_loss_no_weights_${e}epoch
        logger.wandb.tags=[best_model,3cl_rlsa_vh,sauvola,3cl,pre-training,unet,dataset_rlsa_vh_3cl_1,CB55-ssl-set,${e}-epochs]
        logger.wandb.group=pre-training-rlsa-vh-3cl-${e}-ep"
    python run.py ${params}
  done
done
