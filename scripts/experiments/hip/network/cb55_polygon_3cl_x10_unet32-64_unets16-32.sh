#!/usr/bin/env bash

set -e

devices="[0,1,2,3]"
#
#for i in {0..9}; do
#  params="experiment=cb55_polygon_3cl_train_20_run_unet32.yaml
#      trainer.devices=${devices}
#      trainer.max_epochs=100
#
#      mode=hip_networks.yaml
#      name=baseline_cb55_polygon_3cl_unet32_loss_no_weights_100_ep
#      logger.wandb.project=hip_networks
#      logger.wandb.tags=[unet32,polygon,cb55,3-classes,100-epochs,no-weights,baseline]
#      logger.wandb.group=baseline-polygon-cb55"
##    echo ${params}
#      python run.py ${params}
#done
#
#for i in {0..9}; do
#  params="experiment=cb55_polygon_3cl_train_20_run_unet64.yaml
#      trainer.devices=${devices}
#      trainer.max_epochs=100
#      mode=hip_networks.yaml
#      name=baseline_cb55_polygon_3cl_unet64_loss_no_weights_100_ep
#      logger.wandb.project=hip_networks
#      logger.wandb.tags=[unet64,polygon,cb55,3-classes,100-epochs,no-weights,baseline]
#      logger.wandb.group=baseline-polygon-cb55"
##    echo ${params}
#      python run.py ${params}
#done
#
#for i in {0..9}; do
#  params="experiment=cb55_polygon_3cl_train_20_run_unet.yaml
#      trainer.devices=${devices}
#      trainer.max_epochs=100
#      mode=hip_networks.yaml
#      model.header.features=32
#      model.backbone.features_start=32
#      name=baseline_cb55_polygon_3cl_unet-32s_loss_no_weights_100_ep
#      logger.wandb.project=hip_networks
#      logger.wandb.tags=[unet-32s,polygon,cb55,3-classes,100-epochs,no-weights,baseline]
#      logger.wandb.group=baseline-polygon-cb55"
##    echo ${params}
#      python run.py ${params}
#done
#
for i in {0..1}; do
  params="experiment=cb55_polygon_3cl_train_20_run_unet.yaml
      trainer.devices=${devices}
      trainer.max_epochs=100
      mode=hip_networks.yaml
      model.header.features=16
      model.backbone.features_start=16
      name=baseline_cb55_polygon_3cl_unet-16s_loss_no_weights_100_ep
      logger.wandb.project=hip_networks
      logger.wandb.tags=[unet-16s,polygon,cb55,3-classes,100-epochs,no-weights,baseline]
      logger.wandb.group=baseline-polygon-cb55"
#    echo ${params}
      python run.py ${params}
done
#
#for i in {0..9}; do
#  params="experiment=cb55_polygon_3cl_train_20_run_unet16.yaml
#      trainer.devices=${devices}
#      trainer.max_epochs=100
#      mode=hip_networks.yaml
#      name=baseline_cb55_polygon_3cl_unet16_loss_no_weights_100_ep
#      logger.wandb.project=hip_networks
#      logger.wandb.tags=[unet16,polygon,cb55,3-classes,100-epochs,no-weights,baseline]
#      logger.wandb.group=baseline-polygon-cb55"
##    echo ${params}
#      python run.py ${params}
#done
#
#for i in {0..9}; do
#  params="experiment=cb55_polygon_3cl_train_20_run_unet.yaml
#      trainer.devices=${devices}
#      trainer.max_epochs=100
#      mode=hip_networks.yaml
#      name=baseline_cb55_polygon_3cl_unet_loss_no_weights_100_ep
#      logger.wandb.project=hip_networks
#      logger.wandb.tags=[unet,polygon,cb55,3-classes,100-epochs,no-weights,baseline]
#      logger.wandb.group=baseline-polygon-cb55"
##    echo ${params}
#      python run.py ${params}
#done
