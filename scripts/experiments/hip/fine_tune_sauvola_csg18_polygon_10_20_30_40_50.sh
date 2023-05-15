#!/usr/bin/env bash

set -e

weights_10=("/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg18_sauvola_unet_loss_no_weights_10epoch/2023-03-21/08-07-24/checkpoints/epoch\=7/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg18_sauvola_unet_loss_no_weights_10epoch/2023-03-21/08-11-21/checkpoints/epoch\=9/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg18_sauvola_unet_loss_no_weights_10epoch/2023-03-21/08-15-19/checkpoints/epoch\=8/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg18_sauvola_unet_loss_no_weights_10epoch/2023-03-21/08-19-15/checkpoints/epoch\=8/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg18_sauvola_unet_loss_no_weights_10epoch/2023-03-21/08-23-06/checkpoints/epoch\=9/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg18_sauvola_unet_loss_no_weights_10epoch/2023-03-21/08-26-58/checkpoints/epoch\=8/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg18_sauvola_unet_loss_no_weights_10epoch/2023-03-21/08-30-49/checkpoints/epoch\=7/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg18_sauvola_unet_loss_no_weights_10epoch/2023-03-21/08-34-39/checkpoints/epoch\=8/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg18_sauvola_unet_loss_no_weights_10epoch/2023-03-21/08-38-29/checkpoints/epoch\=9/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg18_sauvola_unet_loss_no_weights_10epoch/2023-03-21/08-42-23/checkpoints/epoch\=9/backbone.pth")

devices="[4,5,6,7]"

for j in "${!weights_10[@]}"; do
  params="experiment=fine_tune_csg18_polygon_unet.yaml
          trainer.devices=${devices}
          mode=hip.yaml
          +model.backbone.layers_to_load=[layers.0,layers.1,layers.2,layers.3,layers.4]
          +model.backbone.path_to_weights=${weights_10[$j]}
          name=FT_encoder_sauvola_csg18_polygon_unet_loss_no_weights_10pt_100e_960_1440
          logger.wandb.tags=[unet,csg18,polygon,3-classes,fine-tune,100-epochs,no-weights,sauvola,10-epoch-pt,with_header]
          logger.wandb.project=hip
          logger.wandb.group=fine-tune-sauvola-3cl-polygon-csg18-10pt"
  python run.py ${params}
  #    echo ${params}
done

weights_20=("/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg18_sauvola_unet_loss_no_weights_20epoch/2023-03-21/08-46-13/checkpoints/epoch\=15/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg18_sauvola_unet_loss_no_weights_20epoch/2023-03-21/08-52-38/checkpoints/epoch\=19/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg18_sauvola_unet_loss_no_weights_20epoch/2023-03-21/08-59-13/checkpoints/epoch\=14/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg18_sauvola_unet_loss_no_weights_20epoch/2023-03-21/09-05-38/checkpoints/epoch\=14/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg18_sauvola_unet_loss_no_weights_20epoch/2023-03-21/09-11-59/checkpoints/epoch\=15/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg18_sauvola_unet_loss_no_weights_20epoch/2023-03-21/09-18-26/checkpoints/epoch\=12/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg18_sauvola_unet_loss_no_weights_20epoch/2023-03-21/09-24-48/checkpoints/epoch\=18/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg18_sauvola_unet_loss_no_weights_20epoch/2023-03-21/09-31-20/checkpoints/epoch\=18/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg18_sauvola_unet_loss_no_weights_20epoch/2023-03-21/09-37-55/checkpoints/epoch\=15/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg18_sauvola_unet_loss_no_weights_20epoch/2023-03-21/09-44-22/checkpoints/epoch\=17/backbone.pth"
)

for j in "${!weights_20[@]}"; do
  params="experiment=fine_tune_csg18_polygon_unet.yaml
          trainer.devices=${devices}
          mode=hip.yaml
          +model.backbone.layers_to_load=[layers.0,layers.1,layers.2,layers.3,layers.4]
          +model.backbone.path_to_weights=${weights_20[$j]}
          name=FT_encoder_sauvola_csg18_polygon_unet_loss_no_weights_20pt_100e_960_1440
          logger.wandb.tags=[unet,csg18,polygon,3-classes,fine-tune,100-epochs,no-weights,sauvola,20-epoch-pt,with_header]
          logger.wandb.project=hip
          logger.wandb.group=fine-tune-sauvola-3cl-polygon-csg18-20pt"
  python run.py ${params}
  #    echo ${params}
done

weights_30=("/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg18_sauvola_unet_loss_no_weights_30epoch/2023-03-21/09-50-59/checkpoints/epoch\=24/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg18_sauvola_unet_loss_no_weights_30epoch/2023-03-21/09-59-58/checkpoints/epoch\=23/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg18_sauvola_unet_loss_no_weights_30epoch/2023-03-21/10-09-04/checkpoints/epoch\=16/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg18_sauvola_unet_loss_no_weights_30epoch/2023-03-21/10-18-01/checkpoints/epoch\=29/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg18_sauvola_unet_loss_no_weights_30epoch/2023-03-21/10-27-08/checkpoints/epoch\=19/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg18_sauvola_unet_loss_no_weights_30epoch/2023-03-21/10-36-18/checkpoints/epoch\=25/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg18_sauvola_unet_loss_no_weights_30epoch/2023-03-21/10-45-23/checkpoints/epoch\=24/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg18_sauvola_unet_loss_no_weights_30epoch/2023-03-21/10-54-33/checkpoints/epoch\=26/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg18_sauvola_unet_loss_no_weights_30epoch/2023-03-21/11-03-39/checkpoints/epoch\=26/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg18_sauvola_unet_loss_no_weights_30epoch/2023-03-21/11-12-36/checkpoints/epoch\=29/backbone.pth"
)

for j in "${!weights_30[@]}"; do
  params="experiment=fine_tune_csg18_polygon_unet.yaml
          trainer.devices=${devices}
          mode=hip.yaml
          +model.backbone.layers_to_load=[layers.0,layers.1,layers.2,layers.3,layers.4]
          +model.backbone.path_to_weights=${weights_30[$j]}
          name=FT_encoder_sauvola_csg18_polygon_unet_loss_no_weights_30pt_100e_960_1440
          logger.wandb.tags=[unet,csg18,polygon,3-classes,fine-tune,100-epochs,no-weights,sauvola,30-epoch-pt,with_header]
          logger.wandb.project=hip
          logger.wandb.group=fine-tune-sauvola-3cl-polygon-csg18-30pt"
  python run.py ${params}
  #    echo ${params}
done

weights_40=("/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg18_sauvola_unet_loss_no_weights_40epoch/2023-03-21/11-21-42/checkpoints/epoch\=38/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg18_sauvola_unet_loss_no_weights_40epoch/2023-03-21/11-33-12/checkpoints/epoch\=14/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg18_sauvola_unet_loss_no_weights_40epoch/2023-03-21/11-44-34/checkpoints/epoch\=21/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg18_sauvola_unet_loss_no_weights_40epoch/2023-03-21/11-56-06/checkpoints/epoch\=29/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg18_sauvola_unet_loss_no_weights_40epoch/2023-03-21/12-07-25/checkpoints/epoch\=15/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg18_sauvola_unet_loss_no_weights_40epoch/2023-03-21/12-18-48/checkpoints/epoch\=23/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg18_sauvola_unet_loss_no_weights_40epoch/2023-03-21/12-30-23/checkpoints/epoch\=39/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg18_sauvola_unet_loss_no_weights_40epoch/2023-03-21/12-41-50/checkpoints/epoch\=25/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg18_sauvola_unet_loss_no_weights_40epoch/2023-03-21/12-53-20/checkpoints/epoch\=20/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg18_sauvola_unet_loss_no_weights_40epoch/2023-03-21/13-04-48/checkpoints/epoch\=35/backbone.pth"
)

for j in "${!weights_40[@]}"; do
  params="experiment=fine_tune_csg18_polygon_unet.yaml
          trainer.devices=${devices}
          mode=hip.yaml
          +model.backbone.layers_to_load=[layers.0,layers.1,layers.2,layers.3,layers.4]
          +model.backbone.path_to_weights=${weights_40[$j]}
          name=FT_encoder_sauvola_csg18_polygon_unet_loss_no_weights_40pt_100e_960_1440
          logger.wandb.tags=[unet,csg18,polygon,3-classes,fine-tune,100-epochs,no-weights,sauvola,40-epoch-pt,with_header]
          logger.wandb.project=hip
          logger.wandb.group=fine-tune-sauvola-3cl-polygon-csg18-40pt"
  python run.py ${params}
  #    echo ${params}
done

weights_50=("/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg18_sauvola_unet_loss_no_weights_50epoch/2023-03-21/13-16-31/checkpoints/epoch\=20/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg18_sauvola_unet_loss_no_weights_50epoch/2023-03-21/13-30-28/checkpoints/epoch\=26/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg18_sauvola_unet_loss_no_weights_50epoch/2023-03-21/13-44-40/checkpoints/epoch\=36/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg18_sauvola_unet_loss_no_weights_50epoch/2023-03-21/13-58-36/checkpoints/epoch\=40/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg18_sauvola_unet_loss_no_weights_50epoch/2023-03-21/14-12-39/checkpoints/epoch\=38/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg18_sauvola_unet_loss_no_weights_50epoch/2023-03-21/14-26-52/checkpoints/epoch\=44/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg18_sauvola_unet_loss_no_weights_50epoch/2023-03-21/14-40-50/checkpoints/epoch\=27/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg18_sauvola_unet_loss_no_weights_50epoch/2023-03-21/14-54-46/checkpoints/epoch\=42/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg18_sauvola_unet_loss_no_weights_50epoch/2023-03-21/15-08-41/checkpoints/epoch\=37/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg18_sauvola_unet_loss_no_weights_50epoch/2023-03-21/15-22-36/checkpoints/epoch\=35/backbone.pth"
)

for j in "${!weights_50[@]}"; do
  params="experiment=fine_tune_csg18_polygon_unet.yaml
          trainer.devices=${devices}
          mode=hip.yaml
          +model.backbone.layers_to_load=[layers.0,layers.1,layers.2,layers.3,layers.4]
          +model.backbone.path_to_weights=${weights_50[$j]}
          name=FT_encoder_sauvola_csg18_polygon_unet_loss_no_weights_50pt_100e_960_1440
          logger.wandb.tags=[unet,csg18,polygon,3-classes,fine-tune,100-epochs,no-weights,sauvola,50-epoch-pt,with_header]
          logger.wandb.project=hip
          logger.wandb.group=fine-tune-sauvola-3cl-polygon-csg18-50pt"
  python run.py ${params}
  #    echo ${params}
done
