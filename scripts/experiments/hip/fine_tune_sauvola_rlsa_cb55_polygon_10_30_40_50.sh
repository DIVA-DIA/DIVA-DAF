#!/usr/bin/env bash

set -e

weights_10=("/net/research-hisdoc/experiments_lars_paul/lars/experiments/binary_cb55_sauvola_rlsa_unet_loss_no_weights_real_10epochs/2023-01-18/19-29-53/checkpoints/epoch\=9/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars/experiments/binary_cb55_sauvola_rlsa_unet_loss_no_weights_real_10epochs/2023-01-18/19-39-20/checkpoints/epoch\=9/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars/experiments/binary_cb55_sauvola_rlsa_unet_loss_no_weights_real_10epochs/2023-01-18/19-48-49/checkpoints/epoch\=8/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars/experiments/binary_cb55_sauvola_rlsa_unet_loss_no_weights_real_10epochs/2023-01-18/19-58-09/checkpoints/epoch\=9/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars/experiments/binary_cb55_sauvola_rlsa_unet_loss_no_weights_real_10epochs/2023-01-18/20-07-34/checkpoints/epoch\=7/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars/experiments/binary_cb55_sauvola_rlsa_unet_loss_no_weights_real_10epochs/2023-01-18/20-16-50/checkpoints/epoch\=9/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars/experiments/binary_cb55_sauvola_rlsa_unet_loss_no_weights_real_10epochs/2023-01-18/20-26-22/checkpoints/epoch\=9/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars/experiments/binary_cb55_sauvola_rlsa_unet_loss_no_weights_real_10epochs/2023-01-18/20-35-49/checkpoints/epoch\=8/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars/experiments/binary_cb55_sauvola_rlsa_unet_loss_no_weights_real_10epochs/2023-01-18/20-45-06/checkpoints/epoch\=8/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars/experiments/binary_cb55_sauvola_rlsa_unet_loss_no_weights_real_10epochs/2023-01-18/20-54-22/checkpoints/epoch\=7/backbone.pth")
weights_20=("/net/research-hisdoc/experiments_lars_paul/lars/experiments/binary_cb55_sauvola_rlsa_unet_loss_no_weights_real_20epochs/2023-01-18/21-03-45/checkpoints/epoch\=19/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars/experiments/binary_cb55_sauvola_rlsa_unet_loss_no_weights_real_20epochs/2023-01-18/21-21-02/checkpoints/epoch\=19/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars/experiments/binary_cb55_sauvola_rlsa_unet_loss_no_weights_real_20epochs/2023-01-18/21-38-33/checkpoints/epoch\=13/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars/experiments/binary_cb55_sauvola_rlsa_unet_loss_no_weights_real_20epochs/2023-01-18/21-55-50/checkpoints/epoch\=18/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars/experiments/binary_cb55_sauvola_rlsa_unet_loss_no_weights_real_20epochs/2023-01-18/22-13-17/checkpoints/epoch\=13/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars/experiments/binary_cb55_sauvola_rlsa_unet_loss_no_weights_real_20epochs/2023-01-18/22-30-52/checkpoints/epoch\=13/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars/experiments/binary_cb55_sauvola_rlsa_unet_loss_no_weights_real_20epochs/2023-01-18/22-48-26/checkpoints/epoch\=12/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars/experiments/binary_cb55_sauvola_rlsa_unet_loss_no_weights_real_20epochs/2023-01-18/23-05-53/checkpoints/epoch\=17/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars/experiments/binary_cb55_sauvola_rlsa_unet_loss_no_weights_real_20epochs/2023-01-18/23-23-24/checkpoints/epoch\=15/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars/experiments/binary_cb55_sauvola_rlsa_unet_loss_no_weights_real_20epochs/2023-01-18/23-40-55/checkpoints/epoch\=18/backbone.pth")
weights_30=("/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/binary_cb55_sauvola_rlsa_unet_loss_no_weights_real_30epochs/2023-01-19/10-17-08/checkpoints/epoch\=17/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/binary_cb55_sauvola_rlsa_unet_loss_no_weights_real_30epochs/2023-01-19/10-27-43/checkpoints/epoch\=16/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/binary_cb55_sauvola_rlsa_unet_loss_no_weights_real_30epochs/2023-01-19/10-38-12/checkpoints/epoch\=21/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/binary_cb55_sauvola_rlsa_unet_loss_no_weights_real_30epochs/2023-01-19/10-48-42/checkpoints/epoch\=28/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/binary_cb55_sauvola_rlsa_unet_loss_no_weights_real_30epochs/2023-01-19/10-59-16/checkpoints/epoch\=25/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/binary_cb55_sauvola_rlsa_unet_loss_no_weights_real_30epochs/2023-01-19/11-09-51/checkpoints/epoch\=25/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/binary_cb55_sauvola_rlsa_unet_loss_no_weights_real_30epochs/2023-01-19/11-20-24/checkpoints/epoch\=21/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/binary_cb55_sauvola_rlsa_unet_loss_no_weights_real_30epochs/2023-01-19/11-30-52/checkpoints/epoch\=28/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/binary_cb55_sauvola_rlsa_unet_loss_no_weights_real_30epochs/2023-01-19/11-41-23/checkpoints/epoch\=18/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/binary_cb55_sauvola_rlsa_unet_loss_no_weights_real_30epochs/2023-01-19/11-52-02/checkpoints/epoch\=18/backbone.pth")
weights_40=("/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/binary_cb55_sauvola_rlsa_unet_loss_no_weights_real_40epochs/2023-01-19/12-02-35/checkpoints/epoch\=34/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/binary_cb55_sauvola_rlsa_unet_loss_no_weights_real_40epochs/2023-01-19/12-16-35/checkpoints/epoch\=19/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/binary_cb55_sauvola_rlsa_unet_loss_no_weights_real_40epochs/2023-01-19/12-30-15/checkpoints/epoch\=24/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/binary_cb55_sauvola_rlsa_unet_loss_no_weights_real_40epochs/2023-01-19/12-43-51/checkpoints/epoch\=19/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/binary_cb55_sauvola_rlsa_unet_loss_no_weights_real_40epochs/2023-01-19/12-57-35/checkpoints/epoch\=29/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/binary_cb55_sauvola_rlsa_unet_loss_no_weights_real_40epochs/2023-01-19/13-11-14/checkpoints/epoch\=19/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/binary_cb55_sauvola_rlsa_unet_loss_no_weights_real_40epochs/2023-01-19/13-24-50/checkpoints/epoch\=23/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/binary_cb55_sauvola_rlsa_unet_loss_no_weights_real_40epochs/2023-01-19/13-38-23/checkpoints/epoch\=19/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/binary_cb55_sauvola_rlsa_unet_loss_no_weights_real_40epochs/2023-01-19/13-52-02/checkpoints/epoch\=27/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/binary_cb55_sauvola_rlsa_unet_loss_no_weights_real_40epochs/2023-01-19/14-05-48/checkpoints/epoch\=21/backbone.pth")
weights_50=("/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/binary_cb55_sauvola_rlsa_unet_loss_no_weights_real_50epochs/2023-01-19/14-19-22/checkpoints/epoch\=32/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/binary_cb55_sauvola_rlsa_unet_loss_no_weights_real_50epochs/2023-01-19/14-35-56/checkpoints/epoch\=16/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/binary_cb55_sauvola_rlsa_unet_loss_no_weights_real_50epochs/2023-01-19/14-52-44/checkpoints/epoch\=28/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/binary_cb55_sauvola_rlsa_unet_loss_no_weights_real_50epochs/2023-01-19/15-09-25/checkpoints/epoch\=34/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/binary_cb55_sauvola_rlsa_unet_loss_no_weights_real_50epochs/2023-01-19/15-26-02/checkpoints/epoch\=33/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/binary_cb55_sauvola_rlsa_unet_loss_no_weights_real_50epochs/2023-01-19/15-42-40/checkpoints/epoch\=34/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/binary_cb55_sauvola_rlsa_unet_loss_no_weights_real_50epochs/2023-01-19/15-59-17/checkpoints/epoch\=38/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/binary_cb55_sauvola_rlsa_unet_loss_no_weights_real_50epochs/2023-01-19/16-15-47/checkpoints/epoch\=24/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/binary_cb55_sauvola_rlsa_unet_loss_no_weights_real_50epochs/2023-01-19/16-32-34/checkpoints/epoch\=24/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/binary_cb55_sauvola_rlsa_unet_loss_no_weights_real_50epochs/2023-01-19/16-49-14/checkpoints/epoch\=23/backbone.pth")

devices="[0,1,2,3]"

#for i in ${weights_10[*]}; do
#  params="experiment=fine_tune_cb55_polygon_unet.yaml
#        trainer.devices=${devices}
#        mode=hip.yaml
#        name=FT_sauvola_rlsa_real_cb55_polygon_unet_loss_no_weights_10pt_100e
#        +model.backbone.path_to_weights=${i}
#        logger.wandb.tags=[unet,AB1,3-classes,fine-tune,100-epochs,no-weights,sauvola,binary_rlsa,rlsa_real,10-epoch-pt]
#        logger.wandb.group=fine-tune-sauvola-rlsa-real-10pt"
#  python run.py ${params}
#  #          echo ${params}
#done
#
#for i in ${weights_20[*]}; do
#  params="experiment=fine_tune_cb55_polygon_unet.yaml
#        trainer.devices=${devices}
#        mode=hip.yaml
#        name=FT_sauvola_rlsa_real_cb55_polygon_unet_loss_no_weights_20pt_100e
#        +model.backbone.path_to_weights=${i}
#        logger.wandb.tags=[unet,AB1,3-classes,fine-tune,100-epochs,no-weights,sauvola,binary_rlsa,rlsa_real,20-epoch-pt]
#        logger.wandb.group=fine-tune-sauvola-rlsa-real-20pt"
#  python run.py ${params}
#  #          echo ${params}
#done
#
#for i in ${weights_30[*]}; do
#  params="experiment=fine_tune_cb55_polygon_unet.yaml
#        trainer.devices=${devices}
#        mode=hip.yaml
#        name=FT_sauvola_rlsa_real_cb55_polygon_unet_loss_no_weights_30pt_100e
#        +model.backbone.path_to_weights=${i}
#        logger.wandb.tags=[unet,AB1,3-classes,fine-tune,100-epochs,no-weights,sauvola,binary_rlsa,rlsa_real,30-epoch-pt]
#        logger.wandb.group=fine-tune-sauvola-rlsa-real-30pt"
#  python run.py ${params}
#  #          echo ${params}
#done
#for i in ${weights_40[*]}; do
#  params="experiment=fine_tune_cb55_polygon_unet.yaml
#        trainer.devices=${devices}
#        mode=hip.yaml
#        name=FT_sauvola_rlsa_real_cb55_polygon_unet_loss_no_weights_40pt_100e
#        +model.backbone.path_to_weights=${i}
#        logger.wandb.tags=[unet,AB1,3-classes,fine-tune,100-epochs,no-weights,sauvola,binary_rlsa,rlsa_real,40-epoch-pt]
#        logger.wandb.group=fine-tune-sauvola-rlsa-real-40pt"
#  python run.py ${params}
#  #        echo ${params}
#done

for i in ${weights_50[*]}; do
  params="experiment=fine_tune_cb55_polygon_unet.yaml
        trainer.devices=${devices}
        mode=hip.yaml
        name=FT_sauvola_rlsa_real_cb55_polygon_unet_loss_no_weights_50pt_100e
        +model.backbone.path_to_weights=${i}
        logger.wandb.tags=[unet,AB1,3-classes,fine-tune,100-epochs,no-weights,sauvola,binary_rlsa,rlsa_real,50-epoch-pt]
        logger.wandb.group=fine-tune-sauvola-rlsa-real-50pt"
  python run.py ${params}
  #        echo ${params}
done
