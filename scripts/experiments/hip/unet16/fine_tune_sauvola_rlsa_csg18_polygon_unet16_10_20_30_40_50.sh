#!/usr/bin/env bash

set -e

weights_10=("/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg18_sauvola_rlsa_unet16_loss_no_weights_10epoch/2023-04-20/19-17-15/checkpoints/epoch\=8/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg18_sauvola_rlsa_unet16_loss_no_weights_10epoch/2023-04-20/19-18-59/checkpoints/epoch\=9/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg18_sauvola_rlsa_unet16_loss_no_weights_10epoch/2023-04-20/19-20-41/checkpoints/epoch\=8/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg18_sauvola_rlsa_unet16_loss_no_weights_10epoch/2023-04-20/19-22-25/checkpoints/epoch\=8/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg18_sauvola_rlsa_unet16_loss_no_weights_10epoch/2023-04-20/19-24-07/checkpoints/epoch\=8/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg18_sauvola_rlsa_unet16_loss_no_weights_10epoch/2023-04-20/19-25-52/checkpoints/epoch\=9/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg18_sauvola_rlsa_unet16_loss_no_weights_10epoch/2023-04-20/19-27-37/checkpoints/epoch\=9/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg18_sauvola_rlsa_unet16_loss_no_weights_10epoch/2023-04-20/19-29-23/checkpoints/epoch\=8/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg18_sauvola_rlsa_unet16_loss_no_weights_10epoch/2023-04-20/19-31-06/checkpoints/epoch\=7/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg18_sauvola_rlsa_unet16_loss_no_weights_10epoch/2023-04-20/19-32-49/checkpoints/epoch\=7/backbone.pth"
)
devices="[0,1,2,3]"
#
#for j in "${!weights_10[@]}"; do
#  params="experiment=fine_tune_csg18_polygon_unet.yaml
#          trainer.devices=${devices}
#          mode=hip.yaml
#          +model.backbone.path_to_weights=${weights_10[$j]}
#          name=FT_sauvola_rlsa_csg18_polygon_unet16_loss_no_weights_10pt_100e_1152_1728
#          logger.wandb.tags=[unet16,csg18,polygon,3-classes,fine-tune,100-epochs,no-weights,sauvola_rlsa,10-epoch-pt,with_header]
#          logger.wandb.project=hip
#          logger.wandb.group=fine-tune-sauvola_rlsa-3cl-polygon-csg18-10pt"
#  python run.py ${params}
#  #    echo ${params}
#done

weights_20=("/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg18_sauvola_rlsa_unet16_loss_no_weights_20epoch/2023-04-20/19-34-33/checkpoints/epoch\=14/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg18_sauvola_rlsa_unet16_loss_no_weights_20epoch/2023-04-20/19-37-07/checkpoints/epoch\=14/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg18_sauvola_rlsa_unet16_loss_no_weights_20epoch/2023-04-20/19-39-43/checkpoints/epoch\=15/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg18_sauvola_rlsa_unet16_loss_no_weights_20epoch/2023-04-20/19-42-13/checkpoints/epoch\=19/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg18_sauvola_rlsa_unet16_loss_no_weights_20epoch/2023-04-20/19-44-46/checkpoints/epoch\=17/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg18_sauvola_rlsa_unet16_loss_no_weights_20epoch/2023-04-20/19-47-19/checkpoints/epoch\=15/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg18_sauvola_rlsa_unet16_loss_no_weights_20epoch/2023-04-20/19-49-49/checkpoints/epoch\=19/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg18_sauvola_rlsa_unet16_loss_no_weights_20epoch/2023-04-20/19-52-22/checkpoints/epoch\=13/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg18_sauvola_rlsa_unet16_loss_no_weights_20epoch/2023-04-20/19-54-55/checkpoints/epoch\=18/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg18_sauvola_rlsa_unet16_loss_no_weights_20epoch/2023-04-20/19-57-34/checkpoints/epoch\=16/backbone.pth"
)
#
#for j in "${!weights_20[@]}"; do
#  params="experiment=fine_tune_csg18_polygon_unet.yaml
#          trainer.devices=${devices}
#          mode=hip.yaml
#          +model.backbone.path_to_weights=${weights_20[$j]}
#          name=FT_sauvola_rlsa_csg18_polygon_unet16_loss_no_weights_20pt_100e_1152_1728
#          logger.wandb.tags=[unet16,csg18,polygon,3-classes,fine-tune,100-epochs,no-weights,sauvola_rlsa,20-epoch-pt,with_header]
#          logger.wandb.project=hip
#          logger.wandb.group=fine-tune-sauvola_rlsa-3cl-polygon-csg18-20pt"
#  python run.py ${params}
#  #    echo ${params}
#done

weights_30=("/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg18_sauvola_rlsa_unet16_loss_no_weights_30epoch/2023-04-20/20-00-12/checkpoints/epoch\=25/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg18_sauvola_rlsa_unet16_loss_no_weights_30epoch/2023-04-20/20-03-37/checkpoints/epoch\=19/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg18_sauvola_rlsa_unet16_loss_no_weights_30epoch/2023-04-20/20-06-59/checkpoints/epoch\=17/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg18_sauvola_rlsa_unet16_loss_no_weights_30epoch/2023-04-20/20-10-20/checkpoints/epoch\=19/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg18_sauvola_rlsa_unet16_loss_no_weights_30epoch/2023-04-20/20-13-42/checkpoints/epoch\=26/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg18_sauvola_rlsa_unet16_loss_no_weights_30epoch/2023-04-20/20-17-06/checkpoints/epoch\=26/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg18_sauvola_rlsa_unet16_loss_no_weights_30epoch/2023-04-20/20-20-30/checkpoints/epoch\=26/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg18_sauvola_rlsa_unet16_loss_no_weights_30epoch/2023-04-20/20-23-54/checkpoints/epoch\=28/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg18_sauvola_rlsa_unet16_loss_no_weights_30epoch/2023-04-20/20-27-17/checkpoints/epoch\=20/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg18_sauvola_rlsa_unet16_loss_no_weights_30epoch/2023-04-20/20-30-44/checkpoints/epoch\=16/backbone.pth"
)
#
#for j in "${!weights_30[@]}"; do
#  params="experiment=fine_tune_csg18_polygon_unet.yaml
#          trainer.devices=${devices}
#          mode=hip.yaml
#          +model.backbone.path_to_weights=${weights_30[$j]}
#          name=FT_sauvola_rlsa_csg18_polygon_unet16_loss_no_weights_30pt_100e_1152_1728
#          logger.wandb.tags=[unet16,csg18,polygon,3-classes,fine-tune,100-epochs,no-weights,sauvola_rlsa,30-epoch-pt,with_header]
#          logger.wandb.project=hip
#          logger.wandb.group=fine-tune-sauvola_rlsa-3cl-polygon-csg18-30pt"
#  python run.py ${params}
#  #    echo ${params}
#done

weights_40=("/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg18_sauvola_rlsa_unet16_loss_no_weights_40epoch/2023-04-20/20-34-06/checkpoints/epoch\=37/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg18_sauvola_rlsa_unet16_loss_no_weights_40epoch/2023-04-20/20-38-18/checkpoints/epoch\=25/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg18_sauvola_rlsa_unet16_loss_no_weights_40epoch/2023-04-20/20-42-33/checkpoints/epoch\=24/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg18_sauvola_rlsa_unet16_loss_no_weights_40epoch/2023-04-20/20-46-47/checkpoints/epoch\=29/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg18_sauvola_rlsa_unet16_loss_no_weights_40epoch/2023-04-20/20-51-00/checkpoints/epoch\=29/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg18_sauvola_rlsa_unet16_loss_no_weights_40epoch/2023-04-20/20-55-13/checkpoints/epoch\=39/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg18_sauvola_rlsa_unet16_loss_no_weights_40epoch/2023-04-20/20-59-25/checkpoints/epoch\=30/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg18_sauvola_rlsa_unet16_loss_no_weights_40epoch/2023-04-20/21-03-39/checkpoints/epoch\=24/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg18_sauvola_rlsa_unet16_loss_no_weights_40epoch/2023-04-20/21-07-55/checkpoints/epoch\=23/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg18_sauvola_rlsa_unet16_loss_no_weights_40epoch/2023-04-20/21-12-13/checkpoints/epoch\=28/backbone.pth"
)
#
#for j in "${!weights_40[@]}"; do
#  params="experiment=fine_tune_csg18_polygon_unet.yaml
#          trainer.devices=${devices}
#          mode=hip.yaml
#          +model.backbone.path_to_weights=${weights_40[$j]}
#          name=FT_sauvola_rlsa_csg18_polygon_unet16_loss_no_weights_40pt_100e_1152_1728
#          logger.wandb.tags=[unet16,csg18,polygon,3-classes,fine-tune,100-epochs,no-weights,sauvola_rlsa,40-epoch-pt,with_header]
#          logger.wandb.project=hip
#          logger.wandb.group=fine-tune-sauvola_rlsa-3cl-polygon-csg18-40pt"
#  python run.py ${params}
#  #    echo ${params}
#done

weights_50=("/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg18_sauvola_rlsa_unet16_loss_no_weights_50epoch/2023-04-20/21-16-29/checkpoints/epoch\=37/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg18_sauvola_rlsa_unet16_loss_no_weights_50epoch/2023-04-20/21-21-36/checkpoints/epoch\=20/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg18_sauvola_rlsa_unet16_loss_no_weights_50epoch/2023-04-20/21-26-40/checkpoints/epoch\=30/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg18_sauvola_rlsa_unet16_loss_no_weights_50epoch/2023-04-20/21-31-48/checkpoints/epoch\=31/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg18_sauvola_rlsa_unet16_loss_no_weights_50epoch/2023-04-20/21-36-52/checkpoints/epoch\=38/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg18_sauvola_rlsa_unet16_loss_no_weights_50epoch/2023-04-20/21-42-02/checkpoints/epoch\=30/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg18_sauvola_rlsa_unet16_loss_no_weights_50epoch/2023-04-20/21-47-13/checkpoints/epoch\=43/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg18_sauvola_rlsa_unet16_loss_no_weights_50epoch/2023-04-20/21-52-23/checkpoints/epoch\=41/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg18_sauvola_rlsa_unet16_loss_no_weights_50epoch/2023-04-20/21-57-31/checkpoints/epoch\=34/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg18_sauvola_rlsa_unet16_loss_no_weights_50epoch/2023-04-20/22-02-37/checkpoints/epoch\=39/backbone.pth"
)

for j in "${!weights_50[@]}"; do
  params="experiment=fine_tune_csg18_polygon_unet16.yaml
          trainer.devices=${devices}
          mode=hip.yaml
          +model.backbone.layers_to_load=[conv1,conv2,conv3,conv4,bottleneck]
          +model.backbone.path_to_weights=${weights_50[$j]}
          name=FT_encoder_sauvola_rlsa_csg18_polygon_unet16_loss_no_weights_50pt_100e_1152_1728
          logger.wandb.tags=[unet16,csg18,polygon,3-classes,fine-tune,100-epochs,no-weights,sauvola_rlsa,50-epoch-pt,with_header]
          logger.wandb.project=hip
          logger.wandb.group=fine-tune-sauvola_rlsa-3cl-polygon-csg18-50pt"
  python run.py ${params}
  #    echo ${params}
done
