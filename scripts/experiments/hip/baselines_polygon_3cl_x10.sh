#!/usr/bin/env bash

set -e

devices="[0,1,2,3]"

for i in {0..9}; do
  params="experiment=cb55_polygon_3cl_train_20_run_unet.yaml
      trainer.devices=${devices}
      trainer.max_epochs=100
      mode=hip.yaml
      name=baseline_cb55_polygon_3cl_unet_loss_no_weights_100_ep
      logger.wandb.project=hip
      logger.wandb.tags=[unet,polygon,cb55,3-classes,100-epochs,no-weights,baseline]
      logger.wandb.group=baseline-polygon-cb55"
#    echo ${params}
      python run.py ${params}
done

for i in {0..9}; do
  params="experiment=cb55_polygon_3cl_train_20_run_unet.yaml
      mode=hip.yaml
      trainer.devices=${devices}
      trainer.max_epochs=100
      datamodule.data_dir=/net/research-hisdoc/datasets/semantic_segmentation/datasets/polygon_gt/CSG18/960_1440
      name=baseline_csg18_polygon_3cl_unet_loss_no_weights_100_ep
      logger.wandb.tags=[unet,polygon,csg18,3-classes,100-epochs,no-weights,baseline]
      logger.wandb.project=hip
      logger.wandb.group=baseline-polygon-csg18"
#    echo ${params}
      python run.py ${params}
done

for i in {0..9}; do
  params="experiment=cb55_polygon_3cl_train_20_run_unet.yaml
      mode=hip.yaml
      trainer.devices=${devices}
      trainer.max_epochs=100
      datamodule.data_dir=/net/research-hisdoc/datasets/semantic_segmentation/datasets/polygon_gt/CSG863/960_1440
      name=baseline_csg863_polygon_3cl_unet_loss_no_weights_100_ep
      logger.wandb.tags=[unet,polygon,csg863,3-classes,100-epochs,no-weights,baseline]
      logger.wandb.project=hip
      logger.wandb.group=baseline-polygon-csg863"
#    echo ${params}
      python run.py ${params}
done
