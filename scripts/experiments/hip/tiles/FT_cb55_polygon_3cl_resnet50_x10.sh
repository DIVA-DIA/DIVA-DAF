#!/usr/bin/env bash

set -e

weights_10=("/net/research-hisdoc/experiments_lars_paul/lars_luca/hip_tiles/PT_cb55_3_fixed_resnet50_40epochs/2023-04-28/14-34-44/checkpoints/epoch\=9/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip_tiles/PT_cb55_3_fixed_resnet50_40epochs/2023-04-28/15-49-39/checkpoints/epoch\=9/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip_tiles/PT_cb55_3_fixed_resnet50_40epochs/2023-04-28/17-05-16/checkpoints/epoch\=9/backbone.pth"
)
weights_20=("/net/research-hisdoc/experiments_lars_paul/lars_luca/hip_tiles/PT_cb55_3_fixed_resnet50_40epochs/2023-04-28/14-34-44/checkpoints/epoch\=19/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip_tiles/PT_cb55_3_fixed_resnet50_40epochs/2023-04-28/15-49-39/checkpoints/epoch\=19/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip_tiles/PT_cb55_3_fixed_resnet50_40epochs/2023-04-28/17-05-16/checkpoints/epoch\=19/backbone.pth"
)
weights_30=("/net/research-hisdoc/experiments_lars_paul/lars_luca/hip_tiles/PT_cb55_3_fixed_resnet50_40epochs/2023-04-28/14-34-44/checkpoints/epoch\=29/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip_tiles/PT_cb55_3_fixed_resnet50_40epochs/2023-04-28/15-49-39/checkpoints/epoch\=29/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip_tiles/PT_cb55_3_fixed_resnet50_40epochs/2023-04-28/17-05-16/checkpoints/epoch\=29/backbone.pth"
)
weights_40=("/net/research-hisdoc/experiments_lars_paul/lars_luca/hip_tiles/PT_cb55_3_fixed_resnet50_40epochs/2023-04-28/14-34-44/checkpoints/epoch\=39/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip_tiles/PT_cb55_3_fixed_resnet50_40epochs/2023-04-28/15-49-39/checkpoints/epoch\=39/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip_tiles/PT_cb55_3_fixed_resnet50_40epochs/2023-04-28/17-05-16/checkpoints/epoch\=39/backbone.pth"
)

devices="[5,6,7,8]"

for i in ${weights_10[*]}; do
  params="experiment=cb55_polygon_3cl_run_resnet50.yaml
      trainer.devices=${devices}
      trainer.max_epochs=100
      mode=hip_tiles.yaml
      +model.backbone.path_to_weights=${i}
      name=FT_cb55_tiles_polygon_3cl_resnet50_loss_no_weights_10_pt_100_ep
      logger.wandb.project=hip-tiles
      logger.wandb.tags=[resnet50,polygon,cb55,3-classes,100-epochs,no-weights,10pt]
      logger.wandb.group=FT-tiles-polygon-cb55"
  #    echo ${params}
  python run.py ${params}
done

for i in ${weights_20[*]}; do
  params="experiment=cb55_polygon_3cl_run_resnet50.yaml
      trainer.devices=${devices}
      trainer.max_epochs=100
      mode=hip_tiles.yaml
      +model.backbone.path_to_weights=${i}
      name=FT_cb55_tiles_polygon_3cl_resnet50_loss_no_weights_20_pt_100_ep
      logger.wandb.project=hip-tiles
      logger.wandb.tags=[resnet50,polygon,cb55,3-classes,100-epochs,no-weights,20pt]
      logger.wandb.group=FT-tiles-polygon-cb55"
  #    echo ${params}
  python run.py ${params}
done

for i in ${weights_30[*]}; do
  params="experiment=cb55_polygon_3cl_run_resnet50.yaml
      trainer.devices=${devices}
      trainer.max_epochs=100
      mode=hip_tiles.yaml
      +model.backbone.path_to_weights=${i}
      name=FT_cb55_tiles_polygon_3cl_resnet50_loss_no_weights_30_pt_100_ep
      logger.wandb.project=hip-tiles
      logger.wandb.tags=[resnet50,polygon,cb55,3-classes,100-epochs,no-weights_10,30pt]
      logger.wandb.group=FT-tiles-polygon-cb55"
  #    echo ${params}
  python run.py ${params}
done

for i in ${weights_40[*]}; do
  params="experiment=cb55_polygon_3cl_run_resnet50.yaml
      trainer.devices=${devices}
      trainer.max_epochs=100
      mode=hip_tiles.yaml
      +model.backbone.path_to_weights=${i}
      name=FT_cb55_tiles_polygon_3cl_resnet50_loss_no_weights_40_pt_100_ep
      logger.wandb.project=hip-tiles
      logger.wandb.tags=[resnet50,polygon,cb55,3-classes,100-epochs,no-weights_10,40pt]
      logger.wandb.group=FT-tiles-polygon-cb55"
  #    echo ${params}
  python run.py ${params}
done

