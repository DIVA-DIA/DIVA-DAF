#!/usr/bin/env bash

set -e

devices="[1,2]"

for i in {0..9}; do
  params="experiment=cb55_polygon_3cl_run_resnet50.yaml
      trainer.devices=${devices}
      trainer.max_epochs=100
      mode=hip.yaml
      name=baseline_cb55_polygon_3cl_resnet50_loss_no_weights_100_ep
      logger.wandb.project=hip-tiles
      logger.wandb.tags=[resnet50,polygon,cb55,3-classes,100-epochs,no-weights,baseline]
      logger.wandb.group=baseline-polygon-cb55"
#    echo ${params}
      python run.py ${params}
done
#
#for i in {0..9}; do
#  params="experiment=cb55_polygon_3cl_run_resnet50.yaml
#      mode=hip.yaml
#      trainer.devices=${devices}
#      trainer.max_epochs=100
#      datamodule.data_dir=/net/research-hisdoc/datasets/semantic_segmentation/datasets/polygon_gt/CSG18/960_1440
#      name=baseline_csg18_polygon_3cl_resnet50_loss_no_weights_100_ep
#      logger.wandb.tags=[resnet50,polygon,csg18,3-classes,100-epochs,no-weights,baseline]
#      logger.wandb.project=hip
#      logger.wandb.group=baseline-polygon-csg18"
##    echo ${params}
#      python run.py ${params}
#done
#
#for i in {0..9}; do
#  params="experiment=cb55_polygon_3cl_run_resnet50.yaml
#      mode=hip.yaml
#      trainer.devices=${devices}
#      trainer.max_epochs=100
#      datamodule.data_dir=/net/research-hisdoc/datasets/semantic_segmentation/datasets/polygon_gt/CSG863/960_1440
#      name=baseline_csg863_polygon_3cl_resnet50_loss_no_weights_100_ep
#      logger.wandb.tags=[resnet50,polygon,csg863,3-classes,100-epochs,no-weights,baseline]
#      logger.wandb.project=hip
#      logger.wandb.group=baseline-polygon-csg863"
##    echo ${params}
#      python run.py ${params}
#done
