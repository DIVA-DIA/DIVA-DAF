#!/usr/bin/env bash

set -e

weights_10=("/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_sauvola_rlsa_unet16_loss_no_weights_real_10epochs/2023-04-20/13-10-19/checkpoints/epoch\=9/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_sauvola_rlsa_unet16_loss_no_weights_real_10epochs/2023-04-20/13-12-13/checkpoints/epoch\=9/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_sauvola_rlsa_unet16_loss_no_weights_real_10epochs/2023-04-20/13-14-06/checkpoints/epoch\=9/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_sauvola_rlsa_unet16_loss_no_weights_real_10epochs/2023-04-20/13-15-55/checkpoints/epoch\=9/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_sauvola_rlsa_unet16_loss_no_weights_real_10epochs/2023-04-20/13-17-44/checkpoints/epoch\=9/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_sauvola_rlsa_unet16_loss_no_weights_real_10epochs/2023-04-20/13-19-36/checkpoints/epoch\=8/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_sauvola_rlsa_unet16_loss_no_weights_real_10epochs/2023-04-20/13-21-28/checkpoints/epoch\=8/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_sauvola_rlsa_unet16_loss_no_weights_real_10epochs/2023-04-20/13-23-20/checkpoints/epoch\=9/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_sauvola_rlsa_unet16_loss_no_weights_real_10epochs/2023-04-20/13-25-11/checkpoints/epoch\=9/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_sauvola_rlsa_unet16_loss_no_weights_real_10epochs/2023-04-20/13-27-02/checkpoints/epoch\=9/backbone.pth"
)
weights_20=("/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_sauvola_rlsa_unet16_loss_no_weights_real_20epochs/2023-04-20/13-28-52/checkpoints/epoch\=17/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_sauvola_rlsa_unet16_loss_no_weights_real_20epochs/2023-04-20/13-31-45/checkpoints/epoch\=17/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_sauvola_rlsa_unet16_loss_no_weights_real_20epochs/2023-04-20/13-34-40/checkpoints/epoch\=17/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_sauvola_rlsa_unet16_loss_no_weights_real_20epochs/2023-04-20/13-37-33/checkpoints/epoch\=18/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_sauvola_rlsa_unet16_loss_no_weights_real_20epochs/2023-04-20/13-40-24/checkpoints/epoch\=17/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_sauvola_rlsa_unet16_loss_no_weights_real_20epochs/2023-04-20/13-43-17/checkpoints/epoch\=16/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_sauvola_rlsa_unet16_loss_no_weights_real_20epochs/2023-04-20/13-46-10/checkpoints/epoch\=15/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_sauvola_rlsa_unet16_loss_no_weights_real_20epochs/2023-04-20/13-49-02/checkpoints/epoch\=18/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_sauvola_rlsa_unet16_loss_no_weights_real_20epochs/2023-04-20/13-51-56/checkpoints/epoch\=17/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_sauvola_rlsa_unet16_loss_no_weights_real_20epochs/2023-04-20/13-54-47/checkpoints/epoch\=18/backbone.pth"
)
weights_30=("/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_sauvola_rlsa_unet16_loss_no_weights_real_30epochs/2023-04-20/13-57-39/checkpoints/epoch\=21/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_sauvola_rlsa_unet16_loss_no_weights_real_30epochs/2023-04-20/14-01-37/checkpoints/epoch\=22/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_sauvola_rlsa_unet16_loss_no_weights_real_30epochs/2023-04-20/14-05-35/checkpoints/epoch\=29/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_sauvola_rlsa_unet16_loss_no_weights_real_30epochs/2023-04-20/14-09-34/checkpoints/epoch\=27/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_sauvola_rlsa_unet16_loss_no_weights_real_30epochs/2023-04-20/14-13-30/checkpoints/epoch\=20/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_sauvola_rlsa_unet16_loss_no_weights_real_30epochs/2023-04-20/14-17-29/checkpoints/epoch\=20/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_sauvola_rlsa_unet16_loss_no_weights_real_30epochs/2023-04-20/14-21-28/checkpoints/epoch\=24/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_sauvola_rlsa_unet16_loss_no_weights_real_30epochs/2023-04-20/14-25-25/checkpoints/epoch\=19/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_sauvola_rlsa_unet16_loss_no_weights_real_30epochs/2023-04-20/14-29-23/checkpoints/epoch\=23/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_sauvola_rlsa_unet16_loss_no_weights_real_30epochs/2023-04-20/14-33-15/checkpoints/epoch\=26/backbone.pth"
)
weights_40=("/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_sauvola_rlsa_unet16_loss_no_weights_real_40epochs/2023-04-20/14-37-10/checkpoints/epoch\=36/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_sauvola_rlsa_unet16_loss_no_weights_real_40epochs/2023-04-20/14-42-13/checkpoints/epoch\=39/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_sauvola_rlsa_unet16_loss_no_weights_real_40epochs/2023-04-20/14-47-16/checkpoints/epoch\=30/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_sauvola_rlsa_unet16_loss_no_weights_real_40epochs/2023-04-20/14-52-14/checkpoints/epoch\=37/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_sauvola_rlsa_unet16_loss_no_weights_real_40epochs/2023-04-20/14-57-16/checkpoints/epoch\=36/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_sauvola_rlsa_unet16_loss_no_weights_real_40epochs/2023-04-20/15-02-16/checkpoints/epoch\=35/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_sauvola_rlsa_unet16_loss_no_weights_real_40epochs/2023-04-20/15-07-13/checkpoints/epoch\=38/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_sauvola_rlsa_unet16_loss_no_weights_real_40epochs/2023-04-20/15-12-11/checkpoints/epoch\=27/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_sauvola_rlsa_unet16_loss_no_weights_real_40epochs/2023-04-20/15-17-11/checkpoints/epoch\=36/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_sauvola_rlsa_unet16_loss_no_weights_real_40epochs/2023-04-20/15-22-14/checkpoints/epoch\=38/backbone.pth"
)
weights_50=("/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_sauvola_rlsa_unet16_loss_no_weights_real_50epochs/2023-04-20/15-27-18/checkpoints/epoch\=43/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_sauvola_rlsa_unet16_loss_no_weights_real_50epochs/2023-04-20/15-33-21/checkpoints/epoch\=48/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_sauvola_rlsa_unet16_loss_no_weights_real_50epochs/2023-04-20/15-39-23/checkpoints/epoch\=38/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_sauvola_rlsa_unet16_loss_no_weights_real_50epochs/2023-04-20/15-45-23/checkpoints/epoch\=48/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_sauvola_rlsa_unet16_loss_no_weights_real_50epochs/2023-04-20/15-51-25/checkpoints/epoch\=34/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_sauvola_rlsa_unet16_loss_no_weights_real_50epochs/2023-04-20/15-57-26/checkpoints/epoch\=44/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_sauvola_rlsa_unet16_loss_no_weights_real_50epochs/2023-04-20/16-03-37/checkpoints/epoch\=44/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_sauvola_rlsa_unet16_loss_no_weights_real_50epochs/2023-04-20/16-09-39/checkpoints/epoch\=47/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_sauvola_rlsa_unet16_loss_no_weights_real_50epochs/2023-04-20/16-15-45/checkpoints/epoch\=41/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_sauvola_rlsa_unet16_loss_no_weights_real_50epochs/2023-04-20/16-21-45/checkpoints/epoch\=43/backbone.pth"
)

devices="[0,1,2,3]"

for i in ${weights_10[*]}; do
  params="experiment=fine_tune_cb55_polygon_unet.yaml
        trainer.devices=${devices}
        mode=hip.yaml
        name=FT_sauvola_rlsa_real_cb55_polygon_unet16_loss_no_weights_10pt_100e
        +model.backbone.path_to_weights=${i}
        logger.wandb.project=hip
        logger.wandb.tags=[unet16,AB1,3-classes,fine-tune,100-epochs,no-weights,sauvola,binary_rlsa,rlsa_real,10-epoch-pt]
        logger.wandb.group=fine-tune-sauvola-rlsa-real-10pt"
  python run.py ${params}
  #          echo ${params}
done

for i in ${weights_20[*]}; do
  params="experiment=fine_tune_cb55_polygon_unet.yaml
        trainer.devices=${devices}
        mode=hip.yaml
        name=FT_sauvola_rlsa_real_cb55_polygon_unet16_loss_no_weights_20pt_100e
        +model.backbone.path_to_weights=${i}
        logger.wandb.project=hip
        logger.wandb.tags=[unet16,AB1,3-classes,fine-tune,100-epochs,no-weights,sauvola,binary_rlsa,rlsa_real,20-epoch-pt]
        logger.wandb.group=fine-tune-sauvola-rlsa-real-20pt"
  python run.py ${params}
  #          echo ${params}
done

for i in ${weights_30[*]}; do
  params="experiment=fine_tune_cb55_polygon_unet.yaml
        trainer.devices=${devices}
        mode=hip.yaml
        name=FT_sauvola_rlsa_real_cb55_polygon_unet16_loss_no_weights_30pt_100e
        +model.backbone.path_to_weights=${i}
        logger.wandb.project=hip
        logger.wandb.tags=[unet16,AB1,3-classes,fine-tune,100-epochs,no-weights,sauvola,binary_rlsa,rlsa_real,30-epoch-pt]
        logger.wandb.group=fine-tune-sauvola-rlsa-real-30pt"
  python run.py ${params}
  #          echo ${params}
done
for i in ${weights_40[*]}; do
  params="experiment=fine_tune_cb55_polygon_unet.yaml
        trainer.devices=${devices}
        mode=hip.yaml
        name=FT_sauvola_rlsa_real_cb55_polygon_unet16_loss_no_weights_40pt_100e
        +model.backbone.path_to_weights=${i}
        logger.wandb.project=hip
        logger.wandb.tags=[unet16,AB1,3-classes,fine-tune,100-epochs,no-weights,sauvola,binary_rlsa,rlsa_real,40-epoch-pt]
        logger.wandb.group=fine-tune-sauvola-rlsa-real-40pt"
  python run.py ${params}
  #        echo ${params}
done

for i in ${weights_50[*]}; do
  params="experiment=fine_tune_cb55_polygon_unet.yaml
        trainer.devices=${devices}
        mode=hip.yaml
        name=FT_sauvola_rlsa_real_cb55_polygon_unet16_loss_no_weights_50pt_100e
        +model.backbone.path_to_weights=${i}
        logger.wandb.project=hip
        logger.wandb.tags=[unet16,AB1,3-classes,fine-tune,100-epochs,no-weights,sauvola,binary_rlsa,rlsa_real,50-epoch-pt]
        logger.wandb.group=fine-tune-sauvola-rlsa-real-50pt"
  python run.py ${params}
  #        echo ${params}
done
