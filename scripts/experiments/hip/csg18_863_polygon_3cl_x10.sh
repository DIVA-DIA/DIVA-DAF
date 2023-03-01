#!/usr/bin/env bash

set -e


devices="[0,1,2,3]"

for i in {0..9}; do
  params="experiment=cb55_polygon_3cl_train_20_run_unet.yaml
      mode=hip.yaml
      trainer.devices=${devices}
      trainer.max_epochs=500
      datamodule.data_dir=/net/research-hisdoc/datasets/semantic_segmentation/datasets/polygon_gt/CSG18
      name=sem_seg_csg18_polygon_3cl_unet_loss_no_weights_500_ep
      logger.wandb.tags=[unet,polygon,csg18,3-classes,500-epochs,no-weights,baseline]
      logger.wandb.project=hip
      logger.wandb.group=baseline-polygon-csg18"
#    echo ${params}
      python run.py ${params}
done

for i in {0..9}; do
  params="experiment=cb55_polygon_3cl_train_20_run_unet.yaml
      mode=hip.yaml
      trainer.devices=${devices}
      trainer.max_epochs=500
      datamodule.data_dir=/net/research-hisdoc/datasets/semantic_segmentation/datasets/polygon_gt/CSG863
      name=sem_seg_csg863_polygon_3cl_unet_loss_no_weights_500_ep
      logger.wandb.tags=[unet,polygon,csg863,3-classes,500-epochs,no-weights,baseline]
      logger.wandb.project=hip
      logger.wandb.group=baseline-polygon-csg863"
#    echo ${params}
      python run.py ${params}
done