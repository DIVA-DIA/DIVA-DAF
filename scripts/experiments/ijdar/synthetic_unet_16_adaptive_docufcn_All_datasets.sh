#!/usr/bin/env bash

set -e

dataset_path=("/net/research-hisdoc/datasets/semantic_segmentation/datasets/sythetic/60/dummy_60training+30validation"
  "/net/research-hisdoc/datasets/semantic_segmentation/datasets/sythetic/60/Papyrus+Carolus_60training+30validation"
  "/net/research-hisdoc/datasets/semantic_segmentation/datasets/sythetic/60/Dummy+Papyrus+Carolus_60training+30validation")
group=("setD"
  "setF"
  "setM")

devices=[0,1,2,3]

for j in {0..4}; do
  for i in "${!group[@]}"; do
    params_unet="experiment=synthetic_DPC_unet.yaml
          trainer.devices=${devices}
          trainer.max_epochs=200
          trainer.precision=32
          mode=ijdar.yaml
          name=PT_sem_seg_synthetic_${group[i]}_unet_loss_no_weights_200ep
          datamodule.data_dir=${dataset_path[i]}
          logger.wandb.tags=[unet,${group[i]},4-classes,pre-training,200-epochs,no-weights,no-init]
          logger.wandb.project=ijdar_controlled
          logger.wandb.group=${group[i]}-pt-200ep-unet"
#    echo ${params_unet}
    python run.py ${params_unet}

    params_unet16="experiment=synthetic_DPC_unet16.yaml
          trainer.devices=${devices}
          trainer.max_epochs=200
          trainer.precision=32
          mode=ijdar.yaml
          name=PT_sem_seg_synthetic_${group[i]}_unet_16_loss_no_weights_200ep
          datamodule.data_dir=${dataset_path[i]}
          logger.wandb.tags=[unet-16,${group[i]},4-classes,pre-training,200-epochs,no-weights,no-init]
          logger.wandb.project=ijdar_controlled
          logger.wandb.group=${group[i]}-pt-200ep-unet-16"
#    echo ${params_unet16}
    python run.py ${params_unet16}

    params_unet_adapt="experiment=synthetic_DPC_adaptive_unet.yaml
          trainer.devices=${devices}
          trainer.max_epochs=200
          trainer.precision=32
          mode=ijdar.yaml
          name=PT_sem_seg_synthetic_${group[i]}_adaptive_unet_loss_no_weights_200ep
          datamodule.data_dir=${dataset_path[i]}
          logger.wandb.tags=[adaptive_unet,${group[i]},4-classes,pre-training,200-epochs,no-weights,no-init]
          logger.wandb.project=ijdar_controlled
          logger.wandb.group=${group[i]}-pt-200ep-adaptive-unet"
#    echo ${params_unet_adapt}
    python run.py ${params_unet_adapt}

    params_docufcn="experiment=synthetic_DPC_doc_ufcn.yaml
          trainer.devices=${devices}
          trainer.max_epochs=200
          trainer.precision=32
          mode=ijdar.yaml
          name=PT_sem_seg_synthetic_${group[i]}_doc_ufcn_loss_no_weights-200ep
          datamodule.data_dir=${dataset_path[i]}
          logger.wandb.tags=[doc_ufcn,${group[i]},4-classes,pre-training,200-epochs,no-weights,no-init]
          logger.wandb.project=ijdar_controlled
          logger.wandb.group=${group[i]}-pt-200ep-doc-ufcn"
#    echo ${params_docufcn}
    python run.py ${params_docufcn}
  done
done
