#!/usr/bin/env bash

set -e

dataset_path=("/net/research-hisdoc/datasets/semantic_segmentation/datasets/sythetic/60/dummy_60training+30validation"
  "/net/research-hisdoc/datasets/semantic_segmentation/datasets/sythetic/60/Papyrus+Carolus_60training+30validation"
  "/net/research-hisdoc/datasets/semantic_segmentation/datasets/sythetic/60/Dummy+Papyrus+Carolus_60training+30validation")
group=("D60-setD"
  "DPC60-setF"
  "PC60-setM")

for j in {0..4}; do
  for i in "${!group[@]}"; do
    params_unet="experiment=synthetic_DPC_unet.yaml
          trainer.devices=[4,5,6,7]
          mode=ijdar.yaml
          name=PT_sem_seg_synthetic_${group[i]}_unet_loss_no_weights_200ep
          datamodule.data_dir=${dataset_path[i]}
          logger.wandb.tags=[unet,${group[i]},4-classes,pre-training,200-epochs,no-weights,no-init]
          logger.wandb.project=ijdar_controlled
          logger.wandb.group=${group[i]}-pt-200ep-unet"
    python run.py ${params_unet}

#    params_unet_adapt="experiment=synthetic_DPC_unet_adapt.yaml
#          trainer.devices=[4,5,6,7]
#          trainer.max_epochs=200
#          name=sem_seg_synthetic_${group[i]}_unet_adapted_loss_no_weights
#          datamodule.data_dir=${dataset_path[i]}
#          logger.wandb.tags=[adapted_unet,${group[i]},4-classes,pre-training,200-epochs,no-weights,no-init]
#          logger.wandb.group=${group[i]}"
#    python run.py ${params_unet_adapt}

#    params_unet16="experiment=synthetic_DPC_unet16.yaml
#          trainer.devices=[4,5,6,7]
#          trainer.max_epochs=200
#          name=sem_seg_synthetic_${group[i]}_unet16_loss_no_weights
#          datamodule.data_dir=${dataset_path[i]}
#          logger.wandb.tags=[unet16_new,${group[i]},4-classes,pre-training,200-epochs,no-weights,no-init]
#          logger.wandb.group=${group[i]}"
#    python run.py ${params_unet16}
#
#    params_unet32="experiment=synthetic_DPC_unet32.yaml
#          trainer.devices=[4,5,6,7]
#          trainer.max_epochs=200
#          name=sem_seg_synthetic_${group[i]}_unet32_loss_no_weights
#          datamodule.data_dir=${dataset_path[i]}
#          logger.wandb.tags=[unet32_new,${group[i]},4-classes,pre-training,200-epochs,no-weights,no-init]
#          logger.wandb.group=${group[i]}"
#    python run.py ${params_unet32}
#
#    params_unet64="experiment=synthetic_DPC_unet64.yaml
#          trainer.devices=[4,5,6,7]
#          trainer.max_epochs=200
#          name=sem_seg_synthetic_${group[i]}_unet64_loss_no_weights
#          datamodule.data_dir=${dataset_path[i]}
#          logger.wandb.tags=[unet64_new,${group[i]},4-classes,pre-training,200-epochs,no-weights,no-init]
#          logger.wandb.group=${group[i]}"
#    python run.py ${params_unet64}
  done
done
