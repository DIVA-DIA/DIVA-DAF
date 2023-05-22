#!/usr/bin/env bash

set -e

weights_10=("/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg18_sauvola_unet16_loss_no_weights_10epoch/2023-04-20/16-27-51/checkpoints/epoch\=9/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg18_sauvola_unet16_loss_no_weights_10epoch/2023-04-20/16-29-30/checkpoints/epoch\=8/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg18_sauvola_unet16_loss_no_weights_10epoch/2023-04-20/16-31-13/checkpoints/epoch\=9/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg18_sauvola_unet16_loss_no_weights_10epoch/2023-04-20/16-32-53/checkpoints/epoch\=9/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg18_sauvola_unet16_loss_no_weights_10epoch/2023-04-20/16-34-32/checkpoints/epoch\=9/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg18_sauvola_unet16_loss_no_weights_10epoch/2023-04-20/16-36-13/checkpoints/epoch\=9/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg18_sauvola_unet16_loss_no_weights_10epoch/2023-04-20/16-37-54/checkpoints/epoch\=8/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg18_sauvola_unet16_loss_no_weights_10epoch/2023-04-20/16-39-34/checkpoints/epoch\=9/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg18_sauvola_unet16_loss_no_weights_10epoch/2023-04-20/16-41-13/checkpoints/epoch\=8/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg18_sauvola_unet16_loss_no_weights_10epoch/2023-04-20/16-42-53/checkpoints/epoch\=9/backbone.pth"
)

devices="[0,1,2,3]"

for j in "${!weights_10[@]}"; do
  params="experiment=fine_tune_csg18_polygon_unet16.yaml
          trainer.devices=${devices}
          mode=hip.yaml
          +model.backbone.layers_to_load=[conv1,conv2,conv3,conv4,bottleneck]
          +model.backbone.path_to_weights=${weights_10[$j]}
          name=FT_encoder_sauvola_csg18_polygon_unet16_loss_no_weights_10pt_100e_960_1440
          logger.wandb.tags=[unet16,csg18,polygon,3-classes,fine-tune,100-epochs,no-weights,sauvola,10-epoch-pt,with_header]
          logger.wandb.project=hip
          logger.wandb.group=fine-tune-sauvola-3cl-polygon-csg18-10pt"
  python run.py ${params}
  #    echo ${params}
done

weights_20=("/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg18_sauvola_unet16_loss_no_weights_20epoch/2023-04-20/16-44-33/checkpoints/epoch\=17/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg18_sauvola_unet16_loss_no_weights_20epoch/2023-04-20/16-47-06/checkpoints/epoch\=18/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg18_sauvola_unet16_loss_no_weights_20epoch/2023-04-20/16-49-35/checkpoints/epoch\=18/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg18_sauvola_unet16_loss_no_weights_20epoch/2023-04-20/16-52-09/checkpoints/epoch\=15/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg18_sauvola_unet16_loss_no_weights_20epoch/2023-04-20/16-54-38/checkpoints/epoch\=19/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg18_sauvola_unet16_loss_no_weights_20epoch/2023-04-20/16-57-10/checkpoints/epoch\=19/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg18_sauvola_unet16_loss_no_weights_20epoch/2023-04-20/16-59-37/checkpoints/epoch\=19/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg18_sauvola_unet16_loss_no_weights_20epoch/2023-04-20/17-02-10/checkpoints/epoch\=18/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg18_sauvola_unet16_loss_no_weights_20epoch/2023-04-20/17-04-43/checkpoints/epoch\=17/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg18_sauvola_unet16_loss_no_weights_20epoch/2023-04-20/17-07-13/checkpoints/epoch\=17/backbone.pth"
)

for j in "${!weights_20[@]}"; do
  params="experiment=fine_tune_csg18_polygon_unet16.yaml
          trainer.devices=${devices}
          mode=hip.yaml
          +model.backbone.layers_to_load=[conv1,conv2,conv3,conv4,bottleneck]
          +model.backbone.path_to_weights=${weights_20[$j]}
          name=FT_encoder_sauvola_csg18_polygon_unet16_loss_no_weights_20pt_100e_960_1440
          logger.wandb.tags=[unet16,csg18,polygon,3-classes,fine-tune,100-epochs,no-weights,sauvola,20-epoch-pt,with_header]
          logger.wandb.project=hip
          logger.wandb.group=fine-tune-sauvola-3cl-polygon-csg18-20pt"
  python run.py ${params}
  #    echo ${params}
done

weights_30=("/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg18_sauvola_unet16_loss_no_weights_30epoch/2023-04-20/17-09-44/checkpoints/epoch\=25/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg18_sauvola_unet16_loss_no_weights_30epoch/2023-04-20/17-13-07/checkpoints/epoch\=29/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg18_sauvola_unet16_loss_no_weights_30epoch/2023-04-20/17-16-32/checkpoints/epoch\=23/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg18_sauvola_unet16_loss_no_weights_30epoch/2023-04-20/17-19-54/checkpoints/epoch\=27/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg18_sauvola_unet16_loss_no_weights_30epoch/2023-04-20/17-23-15/checkpoints/epoch\=27/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg18_sauvola_unet16_loss_no_weights_30epoch/2023-04-20/17-26-38/checkpoints/epoch\=25/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg18_sauvola_unet16_loss_no_weights_30epoch/2023-04-20/17-30-01/checkpoints/epoch\=25/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg18_sauvola_unet16_loss_no_weights_30epoch/2023-04-20/17-33-25/checkpoints/epoch\=26/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg18_sauvola_unet16_loss_no_weights_30epoch/2023-04-20/17-36-47/checkpoints/epoch\=29/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg18_sauvola_unet16_loss_no_weights_30epoch/2023-04-20/17-40-10/checkpoints/epoch\=29/backbone.pth"
)

for j in "${!weights_30[@]}"; do
  params="experiment=fine_tune_csg18_polygon_unet16.yaml
          trainer.devices=${devices}
          mode=hip.yaml
          +model.backbone.layers_to_load=[conv1,conv2,conv3,conv4,bottleneck]
          +model.backbone.path_to_weights=${weights_30[$j]}
          name=FT_encoder_sauvola_csg18_polygon_unet16_loss_no_weights_30pt_100e_960_1440
          logger.wandb.tags=[unet16,csg18,polygon,3-classes,fine-tune,100-epochs,no-weights,sauvola,30-epoch-pt,with_header]
          logger.wandb.project=hip
          logger.wandb.group=fine-tune-sauvola-3cl-polygon-csg18-30pt"
  python run.py ${params}
  #    echo ${params}
done

weights_40=("/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg18_sauvola_unet16_loss_no_weights_40epoch/2023-04-20/17-43-36/checkpoints/epoch\=38/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg18_sauvola_unet16_loss_no_weights_40epoch/2023-04-20/17-47-49/checkpoints/epoch\=38/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg18_sauvola_unet16_loss_no_weights_40epoch/2023-04-20/17-52-07/checkpoints/epoch\=36/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg18_sauvola_unet16_loss_no_weights_40epoch/2023-04-20/17-56-18/checkpoints/epoch\=27/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg18_sauvola_unet16_loss_no_weights_40epoch/2023-04-20/18-00-34/checkpoints/epoch\=35/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg18_sauvola_unet16_loss_no_weights_40epoch/2023-04-20/18-04-43/checkpoints/epoch\=32/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg18_sauvola_unet16_loss_no_weights_40epoch/2023-04-20/18-08-59/checkpoints/epoch\=36/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg18_sauvola_unet16_loss_no_weights_40epoch/2023-04-20/18-13-10/checkpoints/epoch\=36/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg18_sauvola_unet16_loss_no_weights_40epoch/2023-04-20/18-17-25/checkpoints/epoch\=32/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg18_sauvola_unet16_loss_no_weights_40epoch/2023-04-20/18-21-41/checkpoints/epoch\=27/backbone.pth"
)

for j in "${!weights_40[@]}"; do
  params="experiment=fine_tune_csg18_polygon_unet16.yaml
          trainer.devices=${devices}
          mode=hip.yaml
          +model.backbone.layers_to_load=[conv1,conv2,conv3,conv4,bottleneck]
          +model.backbone.path_to_weights=${weights_40[$j]}
          name=FT_encoder_sauvola_csg18_polygon_unet16_loss_no_weights_40pt_100e_960_1440
          logger.wandb.tags=[unet16,csg18,polygon,3-classes,fine-tune,100-epochs,no-weights,sauvola,40-epoch-pt,with_header]
          logger.wandb.project=hip
          logger.wandb.group=fine-tune-sauvola-3cl-polygon-csg18-40pt"
  python run.py ${params}
  #    echo ${params}
done

weights_50=("/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg18_sauvola_unet16_loss_no_weights_50epoch/2023-04-20/18-25-56/checkpoints/epoch\=36/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg18_sauvola_unet16_loss_no_weights_50epoch/2023-04-20/18-31-08/checkpoints/epoch\=40/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg18_sauvola_unet16_loss_no_weights_50epoch/2023-04-20/18-36-12/checkpoints/epoch\=44/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg18_sauvola_unet16_loss_no_weights_50epoch/2023-04-20/18-41-19/checkpoints/epoch\=42/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg18_sauvola_unet16_loss_no_weights_50epoch/2023-04-20/18-46-30/checkpoints/epoch\=40/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg18_sauvola_unet16_loss_no_weights_50epoch/2023-04-20/18-51-47/checkpoints/epoch\=39/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg18_sauvola_unet16_loss_no_weights_50epoch/2023-04-20/18-56-50/checkpoints/epoch\=42/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg18_sauvola_unet16_loss_no_weights_50epoch/2023-04-20/19-01-51/checkpoints/epoch\=44/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg18_sauvola_unet16_loss_no_weights_50epoch/2023-04-20/19-07-04/checkpoints/epoch\=38/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg18_sauvola_unet16_loss_no_weights_50epoch/2023-04-20/19-12-11/checkpoints/epoch\=40/backbone.pth"
)

for j in "${!weights_50[@]}"; do
  params="experiment=fine_tune_csg18_polygon_unet16.yaml
          trainer.devices=${devices}
          mode=hip.yaml
          +model.backbone.layers_to_load=[conv1,conv2,conv3,conv4,bottleneck]
          +model.backbone.path_to_weights=${weights_50[$j]}
          name=FT_encoder_sauvola_csg18_polygon_unet16_loss_no_weights_50pt_100e_960_1440
          logger.wandb.tags=[unet16,csg18,polygon,3-classes,fine-tune,100-epochs,no-weights,sauvola,50-epoch-pt,with_header]
          logger.wandb.project=hip
          logger.wandb.group=fine-tune-sauvola-3cl-polygon-csg18-50pt"
  python run.py ${params}
  #    echo ${params}
done
