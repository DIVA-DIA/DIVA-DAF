#!/usr/bin/env bash

set -e

for j in {0..3}; do
  params_unet="experiment=synthetic_DPC_unet.yaml
          trainer.devices=[0,1,2,3]
          trainer.max_epochs=200
          trainer.precision=32
          mode=ijdar.yaml
          name=PT_sem_seg_synthetic_setM_unet_loss_no_weights_200ep
          datamodule.data_dir=/net/research-hisdoc/datasets/semantic_segmentation/datasets/sythetic/60/Dummy+Papyrus+Carolus_60training+30validation
          logger.wandb.tags=[unet,setM,4-classes,pre-training,200-epochs,no-weights,no-init]
          logger.wandb.project=ijdar_controlled
          logger.wandb.group=setM-pt-200ep-unet"
  #    echo ${params_unet}
  python run.py ${params_unet}
done
