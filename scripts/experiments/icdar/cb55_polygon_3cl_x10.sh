#!/usr/bin/env bash

set -e


devices="[0,1,2,3]"

for i in {0..9}; do
  params="experiment=cb55_polygon_3cl_train_20_run_unet.yaml
      trainer.devices=${devices}
      name=sem_seg_cb55_polygon_3cl_unet_loss_no_weights_100_ep
      logger.wandb.tags=[unet,polygon,cb55,3-classes,100-epochs,no-weights,baseline]
      logger.wandb.group=baseline-polygon-cb55"
#    echo ${params}
      python run.py ${params}
done
