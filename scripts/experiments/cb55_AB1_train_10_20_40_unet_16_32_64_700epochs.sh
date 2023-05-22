#!/usr/bin/env bash

set -e
#
#params_unet="experiment=cb55_AB1_train_20_run_unet.yaml
#        trainer.devices=[0,1,2,3]
#        trainer.max_epochs=700"
#
#
#for i in {0..1}
#do
#  python run.py ${params_unet}
#done

dataset_split=("training-10")

epochs_16_32=(100)
epochs_64=(100)
epochs_unet=(100)

for i in "${!dataset_split[@]}"
do
  for j in {1..3}; do
    devices="[1,2,3,4]"
    if [ "${dataset_split[i]}" == "training-10" ]; then
      devices="[1,2]"
    fi
    if [ "${dataset_split[i]}" == "training-20" ]; then
      devices="[1,2]"
    fi

    params_unet="experiment=cb55_AB1_train_20_run_unet.yaml
        trainer.devices=${devices}
        name=sem_seg_baseline_cb55_AB1_loss_no_weights_unet
        logger.wandb.tags=[unet,AB1,4-classes,baseline,${epochs_unet[i]}-epochs,no-weights,${dataset_split[i]}]
        datamodule.train_folder_name=${dataset_split[i]}
        trainer.max_epochs=${epochs_unet[i]}"

    python run.py ${params_unet}

    params_unet_adapted="experiment=cb55_AB1_train_20_run_adapted_unet.yaml
        trainer.devices=${devices}
        name=sem_seg_baseline_cb55_AB1_loss_no_weights_unet_adapted
        logger.wandb.tags=[unet_adapted,AB1,4-classes,baseline,${epochs_unet[i]}-epochs,no-weights,${dataset_split[i]}]
        datamodule.train_folder_name=${dataset_split[i]}
        trainer.max_epochs=${epochs_unet[i]}"

    python run.py ${params_unet_adapted}

    params_unet16="experiment=cb55_AB1_train_20_run_unet16.yaml
        trainer.devices=${devices}
        name=sem_seg_baseline_cb55_AB1_loss_no_weights_unet16
        logger.wandb.tags=[unet16,AB1,4-classes,baseline,${epochs_16_32[i]}-epochs,no-weights,${dataset_split[i]}]
        datamodule.train_folder_name=${dataset_split[i]}
        trainer.max_epochs=${epochs_16_32[i]}"

    python run.py ${params_unet16}
#    echo ${params_unet16}

    params_unet32="experiment=cb55_AB1_train_20_run_unet32.yaml
        trainer.devices=${devices}
        name=sem_seg_baseline_cb55_AB1_loss_no_weights_unet32
        logger.wandb.tags=[unet32,AB1,4-classes,baseline,${epochs_16_32[i]}-epochs,no-weights,${dataset_split[i]}]
         datamodule.train_folder_name=${dataset_split[i]}
        trainer.max_epochs=${epochs_16_32[i]}"

    python run.py ${params_unet32}
#    echo ${params_unet32}

    params_unet64="experiment=cb55_AB1_train_20_run_unet64.yaml
        trainer.devices=${devices}
        name=sem_seg_baseline_cb55_AB1_loss_no_weights_unet64
        logger.wandb.tags=[unet64,AB1,4-classes,baseline,${epochs_64[i]}-epochs,no-weights,${dataset_split[i]}]
        datamodule.train_folder_name=${dataset_split[i]}
        trainer.max_epochs=${epochs_64[i]}"

    python run.py ${params_unet64}
#    echo ${params_unet64}
  done
done
