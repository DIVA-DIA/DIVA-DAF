#!/usr/bin/env bash

set -e

#training_size=("training-20" "training-10" "training-5")
training_size=("training-20" "training-5")

for t in ${training_size[*]}; do
  devices="[0,1,2,3]"
  if [ "${t}" == "training-10" ]; then
    devices="[0,1]"
  fi
  if [ "${t}" == "training-5" ]; then
    devices="[0]"
  fi
  for i in {0..9}; do
    params="experiment=cb55_AB1_3cl_train_20_run_unet.yaml
        trainer.devices=${devices}
        mode=hip.yaml
        datamodule.train_folder_name=${t}
        name=sem_seg_cb55_AB1_3cl_unet_loss_no_weights_100_ep_${t}
        logger.wandb.tags=[unet,AB1,${t},3-classes,100-epochs,no-weights,baseline]
        logger.wandb.project=hip
        logger.wandb.group=baseline-${t}"
#    echo ${params}
        python run.py ${params}
  done
done
