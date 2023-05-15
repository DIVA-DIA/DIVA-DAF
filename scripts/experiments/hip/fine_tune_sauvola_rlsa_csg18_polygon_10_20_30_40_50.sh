#!/usr/bin/env bash

set -e

weights_10=("/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/rlsa_csg18_sauvola_unet_loss_no_weights_10epoch/2023-03-31/13-00-24/checkpoints/epoch\=8/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/rlsa_csg18_sauvola_unet_loss_no_weights_10epoch/2023-03-31/13-04-15/checkpoints/epoch\=6/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/rlsa_csg18_sauvola_unet_loss_no_weights_10epoch/2023-03-31/13-08-06/checkpoints/epoch\=5/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/rlsa_csg18_sauvola_unet_loss_no_weights_10epoch/2023-03-31/13-11-57/checkpoints/epoch\=6/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/rlsa_csg18_sauvola_unet_loss_no_weights_10epoch/2023-03-31/13-15-52/checkpoints/epoch\=6/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/rlsa_csg18_sauvola_unet_loss_no_weights_10epoch/2023-03-31/13-19-43/checkpoints/epoch\=4/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/rlsa_csg18_sauvola_unet_loss_no_weights_10epoch/2023-03-31/13-23-34/checkpoints/epoch\=9/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/rlsa_csg18_sauvola_unet_loss_no_weights_10epoch/2023-03-31/13-27-29/checkpoints/epoch\=4/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/rlsa_csg18_sauvola_unet_loss_no_weights_10epoch/2023-03-31/13-31-24/checkpoints/epoch\=4/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/rlsa_csg18_sauvola_unet_loss_no_weights_10epoch/2023-03-31/13-35-23/checkpoints/epoch\=9/backbone.pth"
)
devices="[1,2,3,4]"

#for j in "${!weights_10[@]}"; do
#  params="experiment=fine_tune_csg18_polygon_unet.yaml
#          trainer.devices=${devices}
#          mode=hip.yaml
#          +model.backbone.path_to_weights=${weights_10[$j]}
#          name=FT_sauvola_rlsa_csg18_polygon_unet_loss_no_weights_10pt_100e_1152_1728
#          logger.wandb.tags=[unet,csg18,polygon,3-classes,fine-tune,100-epochs,no-weights,sauvola_rlsa,10-epoch-pt,with_header]
#          logger.wandb.project=hip
#          logger.wandb.group=fine-tune-sauvola_rlsa-3cl-polygon-csg18-10pt"
#  python run.py ${params}
#  #    echo ${params}
#done

weights_20=("/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/rlsa_csg18_sauvola_unet_loss_no_weights_20epoch/2023-03-31/13-39-14/checkpoints/epoch\=8/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/rlsa_csg18_sauvola_unet_loss_no_weights_20epoch/2023-03-31/13-45-35/checkpoints/epoch\=14/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/rlsa_csg18_sauvola_unet_loss_no_weights_20epoch/2023-03-31/13-51-59/checkpoints/epoch\=14/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/rlsa_csg18_sauvola_unet_loss_no_weights_20epoch/2023-03-31/13-58-21/checkpoints/epoch\=11/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/rlsa_csg18_sauvola_unet_loss_no_weights_20epoch/2023-03-31/14-04-41/checkpoints/epoch\=8/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/rlsa_csg18_sauvola_unet_loss_no_weights_20epoch/2023-03-31/14-11-02/checkpoints/epoch\=19/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/rlsa_csg18_sauvola_unet_loss_no_weights_20epoch/2023-03-31/14-17-24/checkpoints/epoch\=9/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/rlsa_csg18_sauvola_unet_loss_no_weights_20epoch/2023-03-31/14-23-41/checkpoints/epoch\=12/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/rlsa_csg18_sauvola_unet_loss_no_weights_20epoch/2023-03-31/14-30-15/checkpoints/epoch\=15/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/rlsa_csg18_sauvola_unet_loss_no_weights_20epoch/2023-03-31/14-36-36/checkpoints/epoch\=11/backbone.pth"
)

#for j in "${!weights_20[@]}"; do
#  params="experiment=fine_tune_csg18_polygon_unet.yaml
#          trainer.devices=${devices}
#          mode=hip.yaml
#          +model.backbone.path_to_weights=${weights_20[$j]}
#          name=FT_sauvola_rlsa_csg18_polygon_unet_loss_no_weights_20pt_100e_1152_1728
#          logger.wandb.tags=[unet,csg18,polygon,3-classes,fine-tune,100-epochs,no-weights,sauvola_rlsa,20-epoch-pt,with_header]
#          logger.wandb.project=hip
#          logger.wandb.group=fine-tune-sauvola_rlsa-3cl-polygon-csg18-20pt"
#  python run.py ${params}
#  #    echo ${params}
#done

weights_30=("/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/rlsa_csg18_sauvola_unet_loss_no_weights_30epoch/2023-03-31/14-43-06/checkpoints/epoch\=26/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/rlsa_csg18_sauvola_unet_loss_no_weights_30epoch/2023-03-31/14-51-58/checkpoints/epoch\=14/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/rlsa_csg18_sauvola_unet_loss_no_weights_30epoch/2023-03-31/15-00-48/checkpoints/epoch\=16/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/rlsa_csg18_sauvola_unet_loss_no_weights_30epoch/2023-03-31/15-09-44/checkpoints/epoch\=24/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/rlsa_csg18_sauvola_unet_loss_no_weights_30epoch/2023-03-31/15-18-57/checkpoints/epoch\=23/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/rlsa_csg18_sauvola_unet_loss_no_weights_30epoch/2023-03-31/15-27-50/checkpoints/epoch\=21/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/rlsa_csg18_sauvola_unet_loss_no_weights_30epoch/2023-03-31/15-36-43/checkpoints/epoch\=7/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/rlsa_csg18_sauvola_unet_loss_no_weights_30epoch/2023-03-31/15-45-34/checkpoints/epoch\=23/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/rlsa_csg18_sauvola_unet_loss_no_weights_30epoch/2023-03-31/15-54-30/checkpoints/epoch\=26/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/rlsa_csg18_sauvola_unet_loss_no_weights_30epoch/2023-03-31/16-03-36/checkpoints/epoch\=9/backbone.pth"
)

#for j in "${!weights_30[@]}"; do
#  params="experiment=fine_tune_csg18_polygon_unet.yaml
#          trainer.devices=${devices}
#          mode=hip.yaml
#          +model.backbone.path_to_weights=${weights_30[$j]}
#          name=FT_sauvola_rlsa_csg18_polygon_unet_loss_no_weights_30pt_100e_1152_1728
#          logger.wandb.tags=[unet,csg18,polygon,3-classes,fine-tune,100-epochs,no-weights,sauvola_rlsa,30-epoch-pt,with_header]
#          logger.wandb.project=hip
#          logger.wandb.group=fine-tune-sauvola_rlsa-3cl-polygon-csg18-30pt"
#  python run.py ${params}
#  #    echo ${params}
#done

weights_40=("/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/rlsa_csg18_sauvola_unet_loss_no_weights_40epoch/2023-03-31/16-12-26/checkpoints/epoch\=29/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/rlsa_csg18_sauvola_unet_loss_no_weights_40epoch/2023-03-31/16-23-47/checkpoints/epoch\=34/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/rlsa_csg18_sauvola_unet_loss_no_weights_40epoch/2023-03-31/16-35-06/checkpoints/epoch\=33/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/rlsa_csg18_sauvola_unet_loss_no_weights_40epoch/2023-03-31/16-46-28/checkpoints/epoch\=38/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/rlsa_csg18_sauvola_unet_loss_no_weights_40epoch/2023-03-31/16-57-47/checkpoints/epoch\=31/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/rlsa_csg18_sauvola_unet_loss_no_weights_40epoch/2023-03-31/17-09-02/checkpoints/epoch\=37/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/rlsa_csg18_sauvola_unet_loss_no_weights_40epoch/2023-03-31/17-20-41/checkpoints/epoch\=35/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/rlsa_csg18_sauvola_unet_loss_no_weights_40epoch/2023-03-31/17-32-21/checkpoints/epoch\=24/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/rlsa_csg18_sauvola_unet_loss_no_weights_40epoch/2023-03-31/17-43-57/checkpoints/epoch\=33/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/rlsa_csg18_sauvola_unet_loss_no_weights_40epoch/2023-03-31/17-55-24/checkpoints/epoch\=36/backbone.pth"
)

for j in "${!weights_40[@]}"; do
  params="experiment=fine_tune_csg18_polygon_unet.yaml
          trainer.devices=${devices}
          mode=hip.yaml
          +model.backbone.path_to_weights=${weights_40[$j]}
          +model.backbone.layers_to_load=[layers.0,layers.1,layers.2,layers.3,layers.4]
          name=FT_encoder_sauvola_rlsa_csg18_polygon_unet_loss_no_weights_40pt_100e
          logger.wandb.tags=[unet,csg18,polygon,3-classes,fine-tune,100-epochs,no-weights,sauvola_rlsa,40-epoch-pt,with_header]
          logger.wandb.project=hip
          logger.wandb.group=fine-tune-sauvola_rlsa-3cl-polygon-csg18-40pt"
  python run.py ${params}
  #    echo ${params}
done

weights_50=("/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/rlsa_csg18_sauvola_unet_loss_no_weights_50epoch/2023-03-31/18-06-58/checkpoints/epoch\=23/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/rlsa_csg18_sauvola_unet_loss_no_weights_50epoch/2023-03-31/18-20-44/checkpoints/epoch\=37/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/rlsa_csg18_sauvola_unet_loss_no_weights_50epoch/2023-03-31/18-34-56/checkpoints/epoch\=30/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/rlsa_csg18_sauvola_unet_loss_no_weights_50epoch/2023-03-31/18-48-53/checkpoints/epoch\=44/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/rlsa_csg18_sauvola_unet_loss_no_weights_50epoch/2023-03-31/19-02-34/checkpoints/epoch\=43/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/rlsa_csg18_sauvola_unet_loss_no_weights_50epoch/2023-03-31/19-16-26/checkpoints/epoch\=31/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/rlsa_csg18_sauvola_unet_loss_no_weights_50epoch/2023-03-31/19-30-16/checkpoints/epoch\=21/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/rlsa_csg18_sauvola_unet_loss_no_weights_50epoch/2023-03-31/19-44-37/checkpoints/epoch\=34/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/rlsa_csg18_sauvola_unet_loss_no_weights_50epoch/2023-03-31/19-58-31/checkpoints/epoch\=29/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/rlsa_csg18_sauvola_unet_loss_no_weights_50epoch/2023-03-31/20-12-24/checkpoints/epoch\=26/backbone.pth"
)

for j in "${!weights_50[@]}"; do
  params="experiment=fine_tune_csg18_polygon_unet.yaml
          trainer.devices=${devices}
          mode=hip.yaml
          +model.backbone.path_to_weights=${weights_50[$j]}
          +model.backbone.layers_to_load=[layers.0,layers.1,layers.2,layers.3,layers.4]
          name=FT_encoder_sauvola_rlsa_csg18_polygon_unet_loss_no_weights_50pt_100e
          logger.wandb.tags=[unet,csg18,polygon,3-classes,fine-tune,100-epochs,no-weights,sauvola_rlsa,50-epoch-pt,with_header]
          logger.wandb.project=hip
          logger.wandb.group=fine-tune-sauvola_rlsa-3cl-polygon-csg18-50pt"
  python run.py ${params}
  #    echo ${params}
done
