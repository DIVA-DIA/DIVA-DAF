#!/usr/bin/env bash

set -e

weights_10=("/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/binary_cb55_otsu_unet_loss_no_weights_10epochs/2023-01-23/12-36-52/checkpoints/epoch\=5/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/binary_cb55_otsu_unet_loss_no_weights_10epochs/2023-01-23/12-41-19/checkpoints/epoch\=7/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/binary_cb55_otsu_unet_loss_no_weights_10epochs/2023-01-23/12-45-43/checkpoints/epoch\=7/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/binary_cb55_otsu_unet_loss_no_weights_10epochs/2023-01-23/12-50-14/checkpoints/epoch\=8/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/binary_cb55_otsu_unet_loss_no_weights_10epochs/2023-01-23/12-54-40/checkpoints/epoch\=4/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/binary_cb55_otsu_unet_loss_no_weights_10epochs/2023-01-23/12-59-11/checkpoints/epoch\=7/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/binary_cb55_otsu_unet_loss_no_weights_10epochs/2023-01-23/13-03-45/checkpoints/epoch\=6/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/binary_cb55_otsu_unet_loss_no_weights_10epochs/2023-01-23/13-08-13/checkpoints/epoch\=8/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/binary_cb55_otsu_unet_loss_no_weights_10epochs/2023-01-23/13-12-45/checkpoints/epoch\=9/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/binary_cb55_otsu_unet_loss_no_weights_10epochs/2023-01-23/13-17-19/checkpoints/epoch\=9/backbone.pth")
weights_20=("/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/binary_cb55_otsu_unet_loss_no_weights_20epochs/2023-01-23/13-21-49/checkpoints/epoch\=19/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/binary_cb55_otsu_unet_loss_no_weights_20epochs/2023-01-23/13-29-24/checkpoints/epoch\=11/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/binary_cb55_otsu_unet_loss_no_weights_20epochs/2023-01-23/13-36-50/checkpoints/epoch\=8/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/binary_cb55_otsu_unet_loss_no_weights_20epochs/2023-01-23/13-44-16/checkpoints/epoch\=19/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/binary_cb55_otsu_unet_loss_no_weights_20epochs/2023-01-23/13-51-51/checkpoints/epoch\=13/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/binary_cb55_otsu_unet_loss_no_weights_20epochs/2023-01-23/13-59-21/checkpoints/epoch\=14/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/binary_cb55_otsu_unet_loss_no_weights_20epochs/2023-01-23/14-06-53/checkpoints/epoch\=14/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/binary_cb55_otsu_unet_loss_no_weights_20epochs/2023-01-23/14-14-27/checkpoints/epoch\=17/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/binary_cb55_otsu_unet_loss_no_weights_20epochs/2023-01-23/14-21-59/checkpoints/epoch\=17/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/binary_cb55_otsu_unet_loss_no_weights_20epochs/2023-01-23/14-29-33/checkpoints/epoch\=13/backbone.pth")
weights_30=("/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/binary_cb55_otsu_unet_loss_no_weights_30epochs/2023-01-23/16-09-35/checkpoints/epoch\=28/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/binary_cb55_otsu_unet_loss_no_weights_30epochs/2023-01-23/16-20-16/checkpoints/epoch\=9/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/binary_cb55_otsu_unet_loss_no_weights_30epochs/2023-01-23/16-30-57/checkpoints/epoch\=25/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/binary_cb55_otsu_unet_loss_no_weights_30epochs/2023-01-23/16-41-39/checkpoints/epoch\=11/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/binary_cb55_otsu_unet_loss_no_weights_30epochs/2023-01-23/16-52-25/checkpoints/epoch\=27/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/binary_cb55_otsu_unet_loss_no_weights_30epochs/2023-01-23/17-03-00/checkpoints/epoch\=13/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/binary_cb55_otsu_unet_loss_no_weights_30epochs/2023-01-23/17-13-38/checkpoints/epoch\=14/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/binary_cb55_otsu_unet_loss_no_weights_30epochs/2023-01-23/17-24-17/checkpoints/epoch\=19/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/binary_cb55_otsu_unet_loss_no_weights_30epochs/2023-01-23/17-34-56/checkpoints/epoch\=25/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/binary_cb55_otsu_unet_loss_no_weights_30epochs/2023-01-23/17-45-36/checkpoints/epoch\=8/backbone.pth")
weights_40=("/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/binary_cb55_otsu_unet_loss_no_weights_40epochs/2023-01-23/17-56-20/checkpoints/epoch\=23/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/binary_cb55_otsu_unet_loss_no_weights_40epochs/2023-01-23/18-09-58/checkpoints/epoch\=14/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/binary_cb55_otsu_unet_loss_no_weights_40epochs/2023-01-23/18-23-40/checkpoints/epoch\=18/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/binary_cb55_otsu_unet_loss_no_weights_40epochs/2023-01-23/18-37-22/checkpoints/epoch\=30/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/binary_cb55_otsu_unet_loss_no_weights_40epochs/2023-01-23/18-51-08/checkpoints/epoch\=34/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/binary_cb55_otsu_unet_loss_no_weights_40epochs/2023-01-23/19-04-46/checkpoints/epoch\=19/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/binary_cb55_otsu_unet_loss_no_weights_40epochs/2023-01-23/19-18-25/checkpoints/epoch\=35/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/binary_cb55_otsu_unet_loss_no_weights_40epochs/2023-01-23/19-32-02/checkpoints/epoch\=26/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/binary_cb55_otsu_unet_loss_no_weights_40epochs/2023-01-23/19-45-41/checkpoints/epoch\=11/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/binary_cb55_otsu_unet_loss_no_weights_40epochs/2023-01-23/19-59-16/checkpoints/epoch\=26/backbone.pth")
weights_50=("/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/binary_cb55_otsu_unet_loss_no_weights_50epochs/2023-01-21/11-06-09/checkpoints/epoch\=40/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/binary_cb55_otsu_unet_loss_no_weights_50epochs/2023-01-21/11-22-35/checkpoints/epoch\=21/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/binary_cb55_otsu_unet_loss_no_weights_50epochs/2023-01-21/11-39-00/checkpoints/epoch\=22/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/binary_cb55_otsu_unet_loss_no_weights_50epochs/2023-01-21/11-55-25/checkpoints/epoch\=16/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/binary_cb55_otsu_unet_loss_no_weights_50epochs/2023-01-21/12-12-00/checkpoints/epoch\=38/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/binary_cb55_otsu_unet_loss_no_weights_50epochs/2023-01-21/12-28-38/checkpoints/epoch\=48/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/binary_cb55_otsu_unet_loss_no_weights_50epochs/2023-01-21/12-45-12/checkpoints/epoch\=38/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/binary_cb55_otsu_unet_loss_no_weights_50epochs/2023-01-21/13-01-53/checkpoints/epoch\=26/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/binary_cb55_otsu_unet_loss_no_weights_50epochs/2023-01-21/13-18-31/checkpoints/epoch\=24/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/binary_cb55_otsu_unet_loss_no_weights_50epochs/2023-01-21/13-35-08/checkpoints/epoch\=26/backbone.pth")

training=("training-20" "training-10" "training-5")

for t in ${training[*]}; do
  devices="[4,5,6,7]"
  if [ "${t}" == "training-10" ]; then
    devices="[6,7]"
  fi
  if [ "${t}" == "training-5" ]; then
    devices="[7]"
  fi

  for j in ${weights_10[*]}; do
    params="experiment=fine_tune_cb55_AB1_train_20_run_unet.yaml
          trainer.devices=${devices}
          datamodule.train_folder_name=${t}
          +model.backbone.path_to_weights=${j}
          name=fine_tune_otsu_cb55_AB1_${t}_unet_loss_no_weights_10pt_100e
          logger.wandb.tags=[unet,AB1,${t},3-classes,fine-tune,100-epochs,no-weights,otsu,binary,10-epoch-pt]
          logger.wandb.group=fine-tune-otsu-10pt-${t}"
    python run.py ${params}
  done

  for j in ${weights_20[*]}; do
    params="experiment=fine_tune_cb55_AB1_train_20_run_unet.yaml
          trainer.devices=${devices}
          datamodule.train_folder_name=${t}
          +model.backbone.path_to_weights=${j}
          name=fine_tune_otsu_cb55_AB1_${t}_unet_loss_no_weights_20pt_100e
          logger.wandb.tags=[unet,AB1,${t},3-classes,fine-tune,100-epochs,no-weights,otsu,binary,20-epoch-pt]
          logger.wandb.group=fine-tune-otsu-20pt-${t}"
    python run.py ${params}
  done
  
  for j in ${weights_30[*]}; do
    params="experiment=fine_tune_cb55_AB1_train_20_run_unet.yaml
          trainer.devices=${devices}
          datamodule.train_folder_name=${t}
          +model.backbone.path_to_weights=${j}
          name=fine_tune_otsu_cb55_AB1_${t}_unet_loss_no_weights_30pt_100e
          logger.wandb.tags=[unet,AB1,${t},3-classes,fine-tune,100-epochs,no-weights,otsu,binary,30-epoch-pt]
          logger.wandb.group=fine-tune-otsu-30pt-${t}"
    python run.py ${params}
  done

  for j in ${weights_40[*]}; do
    params="experiment=fine_tune_cb55_AB1_train_20_run_unet.yaml
          trainer.devices=${devices}
          datamodule.train_folder_name=${t}
          +model.backbone.path_to_weights=${j}
          name=fine_tune_otsu_cb55_AB1_${t}_unet_loss_no_weights_40pt_100e
          logger.wandb.tags=[unet,AB1,${t},3-classes,fine-tune,100-epochs,no-weights,otsu,binary,40-epoch-pt]
          logger.wandb.group=fine-tune-otsu-40pt-${t}"
    python run.py ${params}
  done

  for j in ${weights_50[*]}; do
    params="experiment=fine_tune_cb55_AB1_train_20_run_unet.yaml
          trainer.devices=${devices}
          datamodule.train_folder_name=${t}
          +model.backbone.path_to_weights=${j}
          name=fine_tune_otsu_cb55_AB1_${t}_unet_loss_no_weights_50pt_100e
          logger.wandb.tags=[unet,AB1,${t},3-classes,fine-tune,100-epochs,no-weights,otsu,binary,50-epoch-pt]
          logger.wandb.group=fine-tune-otsu-50pt-${t}"
    python run.py ${params}
  done
done
