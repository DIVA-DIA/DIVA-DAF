#!/usr/bin/env bash

set -e

weights_10=("/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_sauvola_unet_loss_no_weights_10epochs/2023-01-12/00-11-42/checkpoints/epoch\=9/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_sauvola_unet_loss_no_weights_10epochs/2023-01-12/00-14-51/checkpoints/epoch\=9/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_sauvola_unet_loss_no_weights_10epochs/2023-01-12/00-18-13/checkpoints/epoch\=8/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_sauvola_unet_loss_no_weights_10epochs/2023-01-12/00-21-33/checkpoints/epoch\=7/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_sauvola_unet_loss_no_weights_10epochs/2023-01-12/00-24-47/checkpoints/epoch\=8/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_sauvola_unet_loss_no_weights_10epochs/2023-01-12/00-28-00/checkpoints/epoch\=9/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_sauvola_unet_loss_no_weights_10epochs/2023-01-12/00-31-14/checkpoints/epoch\=5/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_sauvola_unet_loss_no_weights_10epochs/2023-01-12/00-34-31/checkpoints/epoch\=7/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_sauvola_unet_loss_no_weights_10epochs/2023-01-12/00-37-49/checkpoints/epoch\=7/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_sauvola_unet_loss_no_weights_10epochs/2023-01-12/00-40-56/checkpoints/epoch\=9/backbone.pth")
weights_20=("/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_sauvola_unet_loss_no_weights_20epochs/2023-01-12/00-44-11/checkpoints/epoch\=13/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_sauvola_unet_loss_no_weights_20epochs/2023-01-12/00-49-39/checkpoints/epoch\=19/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_sauvola_unet_loss_no_weights_20epochs/2023-01-12/00-55-11/checkpoints/epoch\=15/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_sauvola_unet_loss_no_weights_20epochs/2023-01-12/01-00-42/checkpoints/epoch\=18/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_sauvola_unet_loss_no_weights_20epochs/2023-01-12/01-06-10/checkpoints/epoch\=13/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_sauvola_unet_loss_no_weights_20epochs/2023-01-12/01-11-44/checkpoints/epoch\=17/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_sauvola_unet_loss_no_weights_20epochs/2023-01-12/01-17-10/checkpoints/epoch\=18/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_sauvola_unet_loss_no_weights_20epochs/2023-01-12/01-22-47/checkpoints/epoch\=16/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_sauvola_unet_loss_no_weights_20epochs/2023-01-12/01-28-19/checkpoints/epoch\=16/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_sauvola_unet_loss_no_weights_20epochs/2023-01-12/01-33-56/checkpoints/epoch\=17/backbone.pth")
weights_30=("/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_sauvola_unet_loss_no_weights_30epochs/2023-01-12/01-39-13/checkpoints/epoch\=23/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_sauvola_unet_loss_no_weights_30epochs/2023-01-12/01-46-27/checkpoints/epoch\=27/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_sauvola_unet_loss_no_weights_30epochs/2023-01-12/01-53-42/checkpoints/epoch\=29/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_sauvola_unet_loss_no_weights_30epochs/2023-01-12/02-01-10/checkpoints/epoch\=28/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_sauvola_unet_loss_no_weights_30epochs/2023-01-12/02-08-28/checkpoints/epoch\=23/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_sauvola_unet_loss_no_weights_30epochs/2023-01-12/02-15-58/checkpoints/epoch\=16/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_sauvola_unet_loss_no_weights_30epochs/2023-01-12/02-23-15/checkpoints/epoch\=26/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_sauvola_unet_loss_no_weights_30epochs/2023-01-12/02-30-35/checkpoints/epoch\=24/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_sauvola_unet_loss_no_weights_30epochs/2023-01-12/02-38-02/checkpoints/epoch\=18/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_sauvola_unet_loss_no_weights_30epochs/2023-01-12/02-45-21/checkpoints/epoch\=29/backbone.pth")
weights_40=("/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_sauvola_unet_loss_no_weights_40epochs/2023-01-12/02-52-41/checkpoints/epoch\=37/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_sauvola_unet_loss_no_weights_40epochs/2023-01-12/03-02-12/checkpoints/epoch\=35/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_sauvola_unet_loss_no_weights_40epochs/2023-01-12/03-11-33/checkpoints/epoch\=37/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_sauvola_unet_loss_no_weights_40epochs/2023-01-12/03-20-54/checkpoints/epoch\=38/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_sauvola_unet_loss_no_weights_40epochs/2023-01-12/03-30-19/checkpoints/epoch\=26/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_sauvola_unet_loss_no_weights_40epochs/2023-01-12/03-39-27/checkpoints/epoch\=30/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_sauvola_unet_loss_no_weights_40epochs/2023-01-12/03-48-51/checkpoints/epoch\=39/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_sauvola_unet_loss_no_weights_40epochs/2023-01-12/03-58-09/checkpoints/epoch\=35/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_sauvola_unet_loss_no_weights_40epochs/2023-01-12/04-07-33/checkpoints/epoch\=37/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_sauvola_unet_loss_no_weights_40epochs/2023-01-12/04-16-52/checkpoints/epoch\=35/backbone.pth")
weights_50=("/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_sauvola_unet_loss_no_weights_50epochs/2023-01-12/04-26-25/checkpoints/epoch\=39/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_sauvola_unet_loss_no_weights_50epochs/2023-01-12/04-37-53/checkpoints/epoch\=43/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_sauvola_unet_loss_no_weights_50epochs/2023-01-12/04-49-18/checkpoints/epoch\=46/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_sauvola_unet_loss_no_weights_50epochs/2023-01-12/05-00-48/checkpoints/epoch\=45/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_sauvola_unet_loss_no_weights_50epochs/2023-01-12/05-12-04/checkpoints/epoch\=48/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_sauvola_unet_loss_no_weights_50epochs/2023-01-12/05-23-23/checkpoints/epoch\=41/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_sauvola_unet_loss_no_weights_50epochs/2023-01-12/05-34-32/checkpoints/epoch\=35/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_sauvola_unet_loss_no_weights_50epochs/2023-01-12/05-45-50/checkpoints/epoch\=49/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_sauvola_unet_loss_no_weights_50epochs/2023-01-12/05-57-21/checkpoints/epoch\=36/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_sauvola_unet_loss_no_weights_50epochs/2023-01-12/06-08-38/checkpoints/epoch\=39/backbone.pth")

devices="[4,5,6,7]"
#
#for i in ${weights_10[*]}; do
#  params="experiment=fine_tune_cb55_polygon_unet.yaml
#        trainer.devices=${devices}
#        mode=hip.yaml
#        name=FT_sauvola_cb55_polygon_unet_loss_no_weights_10pt_100e
#        +model.backbone.path_to_weights=${i}
#        logger.wandb.project=hip
#        logger.wandb.tags=[unet,AB1,3-classes,fine-tune,100-epochs,no-weights,sauvola,binary,10-epoch-pt]
#        logger.wandb.group=fine-tune-sauvola-10pt"
#  python run.py ${params}
#  #    echo ${params}
#done
#
#for i in ${weights_20[*]}; do
#  params="experiment=fine_tune_cb55_polygon_unet.yaml
#        trainer.devices=${devices}
#        mode=hip.yaml
#        name=FT_sauvola_cb55_polygon_unet_loss_no_weights_20pt_100e
#        +model.backbone.path_to_weights=${i}
#        logger.wandb.project=hip
#        logger.wandb.tags=[unet,AB1,3-classes,fine-tune,100-epochs,no-weights,sauvola,binary,20-epoch-pt]
#        logger.wandb.group=fine-tune-sauvola-20pt"
#  python run.py ${params}
#  #    echo ${params}
#done
#
#for i in ${weights_30[*]}; do
#  params="experiment=fine_tune_cb55_polygon_unet.yaml
#        trainer.devices=${devices}
#        mode=hip.yaml
#        name=FT_sauvola_cb55_polygon_unet_loss_no_weights_30pt_100e
#        +model.backbone.path_to_weights=${i}
#        logger.wandb.project=hip
#        logger.wandb.tags=[unet,AB1,3-classes,fine-tune,100-epochs,no-weights,sauvola,binary,30-epoch-pt]
#        logger.wandb.group=fine-tune-sauvola-30pt"
#  python run.py ${params}
#  #    echo ${params}
#done
#
#for i in ${weights_40[*]}; do
#  params="experiment=fine_tune_cb55_polygon_unet.yaml
#        trainer.devices=${devices}
#        mode=hip.yaml
#        name=FT_sauvola_cb55_polygon_unet_loss_no_weights_40pt_100e
#        +model.backbone.path_to_weights=${i}
#        logger.wandb.project=hip
#        logger.wandb.tags=[unet,AB1,3-classes,fine-tune,100-epochs,no-weights,sauvola,binary,40-epoch-pt]
#        logger.wandb.group=fine-tune-sauvola-40pt"
#  python run.py ${params}
#  #    echo ${params}
#done

for i in ${weights_50[*]}; do
  params="experiment=fine_tune_cb55_polygon_unet.yaml
        trainer.devices=${devices}
        mode=hip.yaml
        name=FT_sauvola_cb55_polygon_unet_loss_no_weights_50pt_100e
        +model.backbone.path_to_weights=${i}
        logger.wandb.project=hip
        logger.wandb.tags=[unet,AB1,3-classes,fine-tune,100-epochs,no-weights,sauvola,binary,50-epoch-pt]
        logger.wandb.group=fine-tune-sauvola-50pt"
  python run.py ${params}
  #    echo ${params}
done
