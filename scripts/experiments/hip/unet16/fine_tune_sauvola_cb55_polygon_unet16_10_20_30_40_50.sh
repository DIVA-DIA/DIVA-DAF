#!/usr/bin/env bash

set -e

weights_10=("/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_sauvola_unet16_loss_no_weights_10epochs/2023-04-20/08-58-42/checkpoints/epoch\=8/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_sauvola_unet16_loss_no_weights_10epochs/2023-04-20/09-00-35/checkpoints/epoch\=9/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_sauvola_unet16_loss_no_weights_10epochs/2023-04-20/09-02-29/checkpoints/epoch\=8/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_sauvola_unet16_loss_no_weights_10epochs/2023-04-20/09-04-24/checkpoints/epoch\=9/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_sauvola_unet16_loss_no_weights_10epochs/2023-04-20/09-06-17/checkpoints/epoch\=9/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_sauvola_unet16_loss_no_weights_10epochs/2023-04-20/09-08-15/checkpoints/epoch\=9/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_sauvola_unet16_loss_no_weights_10epochs/2023-04-20/09-10-06/checkpoints/epoch\=9/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_sauvola_unet16_loss_no_weights_10epochs/2023-04-20/09-11-54/checkpoints/epoch\=9/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_sauvola_unet16_loss_no_weights_10epochs/2023-04-20/09-13-45/checkpoints/epoch\=9/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_sauvola_unet16_loss_no_weights_10epochs/2023-04-20/09-15-35/checkpoints/epoch\=9/backbone.pth")
weights_20=("/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_sauvola_unet16_loss_no_weights_20epochs/2023-04-20/09-17-25/checkpoints/epoch\=18/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_sauvola_unet16_loss_no_weights_20epochs/2023-04-20/09-20-27/checkpoints/epoch\=14/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_sauvola_unet16_loss_no_weights_20epochs/2023-04-20/09-23-24/checkpoints/epoch\=19/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_sauvola_unet16_loss_no_weights_20epochs/2023-04-20/09-26-21/checkpoints/epoch\=19/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_sauvola_unet16_loss_no_weights_20epochs/2023-04-20/09-29-20/checkpoints/epoch\=18/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_sauvola_unet16_loss_no_weights_20epochs/2023-04-20/09-32-16/checkpoints/epoch\=16/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_sauvola_unet16_loss_no_weights_20epochs/2023-04-20/09-35-07/checkpoints/epoch\=19/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_sauvola_unet16_loss_no_weights_20epochs/2023-04-20/09-38-02/checkpoints/epoch\=18/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_sauvola_unet16_loss_no_weights_20epochs/2023-04-20/09-40-59/checkpoints/epoch\=19/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_sauvola_unet16_loss_no_weights_20epochs/2023-04-20/09-43-56/checkpoints/epoch\=19/backbone.pth")
weights_30=("/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_sauvola_unet16_loss_no_weights_30epochs/2023-04-20/09-46-54/checkpoints/epoch\=26/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_sauvola_unet16_loss_no_weights_30epochs/2023-04-20/09-51-03/checkpoints/epoch\=19/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_sauvola_unet16_loss_no_weights_30epochs/2023-04-20/09-55-04/checkpoints/epoch\=18/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_sauvola_unet16_loss_no_weights_30epochs/2023-04-20/09-59-02/checkpoints/epoch\=28/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_sauvola_unet16_loss_no_weights_30epochs/2023-04-20/10-03-04/checkpoints/epoch\=29/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_sauvola_unet16_loss_no_weights_30epochs/2023-04-20/10-07-05/checkpoints/epoch\=27/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_sauvola_unet16_loss_no_weights_30epochs/2023-04-20/10-11-05/checkpoints/epoch\=23/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_sauvola_unet16_loss_no_weights_30epochs/2023-04-20/10-15-06/checkpoints/epoch\=26/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_sauvola_unet16_loss_no_weights_30epochs/2023-04-20/10-19-05/checkpoints/epoch\=24/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_sauvola_unet16_loss_no_weights_30epochs/2023-04-20/10-23-06/checkpoints/epoch\=27/backbone.pth"
)
weights_40=("/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_sauvola_unet16_loss_no_weights_40epochs/2023-04-20/10-27-04/checkpoints/epoch\=38/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_sauvola_unet16_loss_no_weights_40epochs/2023-04-20/10-32-08/checkpoints/epoch\=35/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_sauvola_unet16_loss_no_weights_40epochs/2023-04-20/10-37-14/checkpoints/epoch\=31/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_sauvola_unet16_loss_no_weights_40epochs/2023-04-20/10-42-11/checkpoints/epoch\=37/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_sauvola_unet16_loss_no_weights_40epochs/2023-04-20/10-47-19/checkpoints/epoch\=36/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_sauvola_unet16_loss_no_weights_40epochs/2023-04-20/10-52-22/checkpoints/epoch\=32/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_sauvola_unet16_loss_no_weights_40epochs/2023-04-20/10-57-22/checkpoints/epoch\=34/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_sauvola_unet16_loss_no_weights_40epochs/2023-04-20/11-02-21/checkpoints/epoch\=27/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_sauvola_unet16_loss_no_weights_40epochs/2023-04-20/11-07-27/checkpoints/epoch\=38/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_sauvola_unet16_loss_no_weights_40epochs/2023-04-20/11-12-33/checkpoints/epoch\=38/backbone.pth"
)
weights_50=("/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_sauvola_unet16_loss_no_weights_50epochs/2023-04-20/11-17-42/checkpoints/epoch\=48/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_sauvola_unet16_loss_no_weights_50epochs/2023-04-20/11-23-52/checkpoints/epoch\=48/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_sauvola_unet16_loss_no_weights_50epochs/2023-04-20/11-30-04/checkpoints/epoch\=48/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_sauvola_unet16_loss_no_weights_50epochs/2023-04-20/11-36-18/checkpoints/epoch\=42/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_sauvola_unet16_loss_no_weights_50epochs/2023-04-20/11-42-32/checkpoints/epoch\=47/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_sauvola_unet16_loss_no_weights_50epochs/2023-04-20/11-48-44/checkpoints/epoch\=39/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_sauvola_unet16_loss_no_weights_50epochs/2023-04-20/11-54-52/checkpoints/epoch\=46/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_sauvola_unet16_loss_no_weights_50epochs/2023-04-20/12-01-06/checkpoints/epoch\=40/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_sauvola_unet16_loss_no_weights_50epochs/2023-04-20/12-07-14/checkpoints/epoch\=43/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_sauvola_unet16_loss_no_weights_50epochs/2023-04-20/12-13-27/checkpoints/epoch\=42/backbone.pth"
)

devices="[0,1,2,3]"

for i in ${weights_10[*]}; do
  params="experiment=fine_tune_cb55_polygon_unet.yaml
        trainer.devices=${devices}
        mode=hip.yaml
        name=FT_sauvola_cb55_polygon_unet16_loss_no_weights_10pt_100e
        +model.backbone.path_to_weights=${i}
        logger.wandb.project=hip
        logger.wandb.tags=[unet16,AB1,3-classes,fine-tune,100-epochs,no-weights,sauvola,binary,10-epoch-pt]
        logger.wandb.group=fine-tune-sauvola-10pt"
  python run.py ${params}
  #    echo ${params}
done

for i in ${weights_20[*]}; do
  params="experiment=fine_tune_cb55_polygon_unet.yaml
        trainer.devices=${devices}
        mode=hip.yaml
        name=FT_sauvola_cb55_polygon_unet16_loss_no_weights_20pt_100e
        +model.backbone.path_to_weights=${i}
        logger.wandb.project=hip
        logger.wandb.tags=[unet16,AB1,3-classes,fine-tune,100-epochs,no-weights,sauvola,binary,20-epoch-pt]
        logger.wandb.group=fine-tune-sauvola-20pt"
  python run.py ${params}
  #    echo ${params}
done

for i in ${weights_30[*]}; do
  params="experiment=fine_tune_cb55_polygon_unet.yaml
        trainer.devices=${devices}
        mode=hip.yaml
        name=FT_sauvola_cb55_polygon_unet16_loss_no_weights_30pt_100e
        +model.backbone.path_to_weights=${i}
        logger.wandb.project=hip
        logger.wandb.tags=[unet16,AB1,3-classes,fine-tune,100-epochs,no-weights,sauvola,binary,30-epoch-pt]
        logger.wandb.group=fine-tune-sauvola-30pt"
  python run.py ${params}
  #    echo ${params}
done

for i in ${weights_40[*]}; do
  params="experiment=fine_tune_cb55_polygon_unet.yaml
        trainer.devices=${devices}
        mode=hip.yaml
        name=FT_sauvola_cb55_polygon_unet16_loss_no_weights_40pt_100e
        +model.backbone.path_to_weights=${i}
        logger.wandb.project=hip
        logger.wandb.tags=[unet16,AB1,3-classes,fine-tune,100-epochs,no-weights,sauvola,binary,40-epoch-pt]
        logger.wandb.group=fine-tune-sauvola-40pt"
  python run.py ${params}
  #    echo ${params}
done

for i in ${weights_50[*]}; do
  params="experiment=fine_tune_cb55_polygon_unet.yaml
        trainer.devices=${devices}
        mode=hip.yaml
        name=FT_sauvola_cb55_polygon_unet16_loss_no_weights_50pt_100e
        +model.backbone.path_to_weights=${i}
        logger.wandb.project=hip
        logger.wandb.tags=[unet16,AB1,3-classes,fine-tune,100-epochs,no-weights,sauvola,binary,50-epoch-pt]
        logger.wandb.group=fine-tune-sauvola-50pt"
  python run.py ${params}
  #    echo ${params}
done
