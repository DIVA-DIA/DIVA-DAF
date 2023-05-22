#!/usr/bin/env bash

set -e


devices="[4,5,6,7]"

for i in {0..9}; do
  params="experiment=cb55_polygon_3cl_train_20_run_unet.yaml
      trainer.devices=${devices}
      datamodule.data_dir=/net/research-hisdoc/datasets/semantic_segmentation/datasets/polygon_gt/CSG18
      name=sem_seg_csg18_polygon_3cl_unet_loss_no_weights_100_ep
      logger.wandb.tags=[unet,polygon,csg18,3-classes,100-epochs,no-weights,baseline]
      logger.wandb.group=baseline-polygon-csg18"
#    echo ${params}
      python run.py ${params}
done

for i in {0..9}; do
  params="experiment=cb55_polygon_3cl_train_20_run_unet.yaml
      trainer.devices=${devices}
      datamodule.data_dir=/net/research-hisdoc/datasets/semantic_segmentation/datasets/polygon_gt/CSG863
      name=sem_seg_csg863_polygon_3cl_unet_loss_no_weights_100_ep
      logger.wandb.tags=[unet,polygon,csg863,3-classes,100-epochs,no-weights,baseline]
      logger.wandb.group=baseline-polygon-csg863"
#    echo ${params}
      python run.py ${params}
done