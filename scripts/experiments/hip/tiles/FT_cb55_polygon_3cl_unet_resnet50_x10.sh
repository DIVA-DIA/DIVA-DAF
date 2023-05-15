#!/usr/bin/env bash

set -e

weights=("/net/research-hisdoc/experiments_lars_paul/lars_luca/hip_tiles/PT_cb55_3_fixed_resnet50_50epochs/2023-04-25/12-02-42/checkpoints/epoch\=49/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip_tiles/PT_cb55_3_fixed_resnet50_50epochs/2023-04-25/13-45-24/checkpoints/epoch\=49/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip_tiles/PT_cb55_3_fixed_resnet50_50epochs/2023-04-25/15-28-19/checkpoints/epoch\=49/backbone.pth"
)

devices="[1,2,3,4]"

for i in ${weights[*]}; do
  params="experiment=cb55_polygon_3cl_run_unet_resnet50_bb.yaml
      trainer.devices=${devices}
      trainer.max_epochs=100
      mode=hip_tiles.yaml
      +model.backbone.strict=False
      +model.backbone.prefix=backbone.
      +model.backbone.path_to_weights=${i}
      name=FT_cb55_tiles_polygon_3cl_unet_resnet50_loss_no_weights_100_ep
      logger.wandb.project=hip-tiles
      logger.wandb.tags=[unet-resnet50,polygon,cb55,3-classes,100-epochs,no-weights,baseline]
      logger.wandb.group=FT-tiles-polygon-cb55-unet-resnet50"
  #    echo ${params}
  python run.py ${params}
done
#
#for i in {0..9}; do
#  params="experiment=cb55_polygon_3cl_run_unet_resnet50_bb.yaml
#      trainer.devices=${devices}
#      trainer.max_epochs=100
#      mode=hip_tiles.yaml
#      +model.backbone.strict=False
#      +model.backbone.prefix=backbone.
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
#  params="experiment=cb55_polygon_3cl_run_unet_resnet50_bb.yaml
#      trainer.devices=${devices}
#      trainer.max_epochs=100
#      mode=hip_tiles.yaml
#      +model.backbone.strict=False
#      +model.backbone.prefix=backbone.
#      datamodule.data_dir=/net/research-hisdoc/datasets/semantic_segmentation/datasets/polygon_gt/CSG863/960_1440
#      name=baseline_csg863_polygon_3cl_resnet50_loss_no_weights_100_ep
#      logger.wandb.tags=[resnet50,polygon,csg863,3-classes,100-epochs,no-weights,baseline]
#      logger.wandb.project=hip
#      logger.wandb.group=baseline-polygon-csg863"
##    echo ${params}
#      python run.py ${params}
#done
