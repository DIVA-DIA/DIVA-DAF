#!/usr/bin/env bash

set -e

devices="[1]"

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
weights_50=("/net/research-hisdoc/experiments_lars_paul/lars_luca/hip_tiles/PT_cb55_3_fixed_resnet50_50epochs/2023-04-25/12-02-42/checkpoints/epoch\=49/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip_tiles/PT_cb55_3_fixed_resnet50_50epochs/2023-04-25/13-45-24/checkpoints/epoch\=49/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip_tiles/PT_cb55_3_fixed_resnet50_50epochs/2023-04-25/15-28-19/checkpoints/epoch\=49/backbone.pth"
)

selects=("5" "1")

for s in ${selects[*]}; do
  acc_batch=1

  for i in ${weights_10[*]}; do
    params="experiment=cb55_polygon_3cl_run_unet_resnet50_bb.yaml
      trainer.devices=${devices}
      trainer.max_epochs=100
      trainer.accumulate_grad_batches=${acc_batch}
      mode=hip_tiles.yaml
      +model.backbone.strict=False
      +model.backbone.prefix=backbone.
      +model.backbone.path_to_weights=${i}
      +datamodule.selection_train=${s}
      name=FT_cb55_${s}train_tiles_polygon_3cl_unet_resnet50_loss_no_weights_10pt_100_ep
      logger.wandb.project=hip-tiles
      logger.wandb.tags=[unet-resnet50,polygon,cb55,3-classes,100-epochs,${s}-train,10-pt]
      logger.wandb.group=FT-tiles-polygon-cb55-unet-resnet50"
    #    echo ${params}
    python run.py ${params}
  done

  for i in ${weights_20[*]}; do
    params="experiment=cb55_polygon_3cl_run_unet_resnet50_bb.yaml
      trainer.devices=${devices}
      trainer.max_epochs=100
      trainer.accumulate_grad_batches=${acc_batch}
      mode=hip_tiles.yaml
      +model.backbone.strict=False
      +model.backbone.prefix=backbone.
      +model.backbone.path_to_weights=${i}
      +datamodule.selection_train=${s}
      name=FT_cb55_${s}train_tiles_polygon_3cl_unet_resnet50_loss_no_weights_20pt_100_ep
      logger.wandb.project=hip-tiles
      logger.wandb.tags=[unet-resnet50,polygon,cb55,3-classes,100-epochs,${s}-train,20-pt]
      logger.wandb.group=FT-tiles-polygon-cb55-unet-resnet50"
    #    echo ${params}
    python run.py ${params}
  done

  for i in ${weights_30[*]}; do
    params="experiment=cb55_polygon_3cl_run_unet_resnet50_bb.yaml
      trainer.devices=${devices}
      trainer.max_epochs=100
      trainer.accumulate_grad_batches=${acc_batch}
      mode=hip_tiles.yaml
      +model.backbone.strict=False
      +model.backbone.prefix=backbone.
      +model.backbone.path_to_weights=${i}
      +datamodule.selection_train=${s}
      name=FT_cb55_${s}train_tiles_polygon_3cl_unet_resnet50_loss_no_weights_30pt_100_ep
      logger.wandb.project=hip-tiles
      logger.wandb.tags=[unet-resnet50,polygon,cb55,3-classes,100-epochs,${s}-train,30-pt]
      logger.wandb.group=FT-tiles-polygon-cb55-unet-resnet50"
    #    echo ${params}
    python run.py ${params}
  done

  for i in ${weights_40[*]}; do
    params="experiment=cb55_polygon_3cl_run_unet_resnet50_bb.yaml
      trainer.devices=${devices}
      trainer.max_epochs=100
      trainer.accumulate_grad_batches=${acc_batch}
      mode=hip_tiles.yaml
      +model.backbone.strict=False
      +model.backbone.prefix=backbone.
      +model.backbone.path_to_weights=${i}
      +datamodule.selection_train=${s}
      name=FT_cb55_${s}train_tiles_polygon_3cl_unet_resnet50_loss_no_weights_40pt_100_ep
      logger.wandb.project=hip-tiles
      logger.wandb.tags=[unet-resnet50,polygon,cb55,3-classes,100-epochs,${s}-train,40-pt]
      logger.wandb.group=FT-tiles-polygon-cb55-unet-resnet50"
    #    echo ${params}
    python run.py ${params}
  done

  for i in ${weights_50[*]}; do
    params="experiment=cb55_polygon_3cl_run_unet_resnet50_bb.yaml
      trainer.devices=${devices}
      trainer.max_epochs=100
      trainer.accumulate_grad_batches=${acc_batch}
      mode=hip_tiles.yaml
      +model.backbone.strict=False
      +model.backbone.prefix=backbone.
      +model.backbone.path_to_weights=${i}
      +datamodule.selection_train=${s}
      name=FT_cb55_${s}train_tiles_polygon_3cl_unet_resnet50_loss_no_weights_50pt_100_ep
      logger.wandb.project=hip-tiles
      logger.wandb.tags=[unet-resnet50,polygon,cb55,3-classes,100-epochs,${s}-train,baseline]
      logger.wandb.group=FT-tiles-polygon-cb55-unet-resnet50"
    #    echo ${params}
    python run.py ${params}
  done

  # BASELINE
  for i in {0..9}; do
    params="experiment=cb55_polygon_3cl_run_unet_resnet50_bb.yaml
      trainer.devices=${devices}
      trainer.max_epochs=100
      trainer.accumulate_grad_batches=${acc_batch}
      mode=hip_tiles.yaml
      +datamodule.selection_train=${s}
      name=baseline_cb55_${s}train_polygon_3cl_unet_resnet50_loss_no_weights_100_ep
      logger.wandb.project=hip-tiles
      logger.wandb.tags=[unet_resnet50,polygon,cb55,3-classes,100-epochs,${s}-train,baseline]
      logger.wandb.group=baseline-polygon-cb55-unet-resnet50"
    #    echo ${params}
    python run.py ${params}
  done
done
