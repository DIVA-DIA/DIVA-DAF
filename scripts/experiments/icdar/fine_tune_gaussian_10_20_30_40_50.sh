#!/usr/bin/env bash

set -e

weights_10=("/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_gaussian_unet_loss_no_weights_10epochs/2023-01-17/13-45-26/checkpoints/epoch\=9/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_gaussian_unet_loss_no_weights_10epochs/2023-01-17/13-49-14/checkpoints/epoch\=8/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_gaussian_unet_loss_no_weights_10epochs/2023-01-17/13-52-54/checkpoints/epoch\=7/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_gaussian_unet_loss_no_weights_10epochs/2023-01-17/13-56-50/checkpoints/epoch\=6/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_gaussian_unet_loss_no_weights_10epochs/2023-01-17/14-00-39/checkpoints/epoch\=9/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_gaussian_unet_loss_no_weights_10epochs/2023-01-17/14-04-18/checkpoints/epoch\=6/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_gaussian_unet_loss_no_weights_10epochs/2023-01-17/14-08-04/checkpoints/epoch\=6/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_gaussian_unet_loss_no_weights_10epochs/2023-01-17/14-11-40/checkpoints/epoch\=9/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_gaussian_unet_loss_no_weights_10epochs/2023-01-17/14-15-33/checkpoints/epoch\=7/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_gaussian_unet_loss_no_weights_10epochs/2023-01-17/14-19-10/checkpoints/epoch\=4/backbone.pth")
weights_20=("/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_gaussian_unet_loss_no_weights_20epochs/2023-01-17/14-22-43/checkpoints/epoch\=13/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_gaussian_unet_loss_no_weights_20epochs/2023-01-17/14-28-40/checkpoints/epoch\=14/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_gaussian_unet_loss_no_weights_20epochs/2023-01-17/14-34-33/checkpoints/epoch\=19/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_gaussian_unet_loss_no_weights_20epochs/2023-01-17/14-40-48/checkpoints/epoch\=9/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_gaussian_unet_loss_no_weights_20epochs/2023-01-17/14-46-35/checkpoints/epoch\=10/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_gaussian_unet_loss_no_weights_20epochs/2023-01-17/14-52-16/checkpoints/epoch\=9/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_gaussian_unet_loss_no_weights_20epochs/2023-01-17/14-59-01/checkpoints/epoch\=15/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_gaussian_unet_loss_no_weights_20epochs/2023-01-17/15-06-00/checkpoints/epoch\=14/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_gaussian_unet_loss_no_weights_20epochs/2023-01-17/15-12-39/checkpoints/epoch\=12/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_gaussian_unet_loss_no_weights_20epochs/2023-01-17/15-19-24/checkpoints/epoch\=18/backbone.pth")
weights_30=("/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_gaussian_unet_loss_no_weights_30epochs/2023-01-17/15-26-11/checkpoints/epoch\=17/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_gaussian_unet_loss_no_weights_30epochs/2023-01-17/15-34-41/checkpoints/epoch\=14/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_gaussian_unet_loss_no_weights_30epochs/2023-01-17/15-43-08/checkpoints/epoch\=21/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_gaussian_unet_loss_no_weights_30epochs/2023-01-17/15-51-12/checkpoints/epoch\=27/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_gaussian_unet_loss_no_weights_30epochs/2023-01-17/15-59-20/checkpoints/epoch\=24/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_gaussian_unet_loss_no_weights_30epochs/2023-01-17/16-07-12/checkpoints/epoch\=11/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_gaussian_unet_loss_no_weights_30epochs/2023-01-17/16-14-55/checkpoints/epoch\=26/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_gaussian_unet_loss_no_weights_30epochs/2023-01-17/16-23-13/checkpoints/epoch\=13/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_gaussian_unet_loss_no_weights_30epochs/2023-01-17/16-31-12/checkpoints/epoch\=21/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_gaussian_unet_loss_no_weights_30epochs/2023-01-17/16-39-14/checkpoints/epoch\=13/backbone.pth")
weights_40=("/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_gaussian_unet_loss_no_weights_40epochs/2023-01-17/16-47-22/checkpoints/epoch\=31/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_gaussian_unet_loss_no_weights_40epochs/2023-01-17/16-57-27/checkpoints/epoch\=17/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_gaussian_unet_loss_no_weights_40epochs/2023-01-17/17-07-41/checkpoints/epoch\=14/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_gaussian_unet_loss_no_weights_40epochs/2023-01-17/17-17-58/checkpoints/epoch\=35/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_gaussian_unet_loss_no_weights_40epochs/2023-01-17/17-27-54/checkpoints/epoch\=25/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_gaussian_unet_loss_no_weights_40epochs/2023-01-17/17-38-13/checkpoints/epoch\=31/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_gaussian_unet_loss_no_weights_40epochs/2023-01-17/17-48-08/checkpoints/epoch\=17/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_gaussian_unet_loss_no_weights_40epochs/2023-01-17/17-58-22/checkpoints/epoch\=14/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_gaussian_unet_loss_no_weights_40epochs/2023-01-17/18-08-29/checkpoints/epoch\=38/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_gaussian_unet_loss_no_weights_40epochs/2023-01-17/18-18-42/checkpoints/epoch\=39/backbone.pth")
weights_50=("/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_gaussian_unet_loss_no_weights_50epochs/2023-01-17/18-29-07/checkpoints/epoch\=48/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_gaussian_unet_loss_no_weights_50epochs/2023-01-17/18-41-37/checkpoints/epoch\=23/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_gaussian_unet_loss_no_weights_50epochs/2023-01-17/18-54-01/checkpoints/epoch\=36/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_gaussian_unet_loss_no_weights_50epochs/2023-01-17/19-06-21/checkpoints/epoch\=30/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_gaussian_unet_loss_no_weights_50epochs/2023-01-17/19-18-51/checkpoints/epoch\=36/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_gaussian_unet_loss_no_weights_50epochs/2023-01-17/19-31-19/checkpoints/epoch\=17/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_gaussian_unet_loss_no_weights_50epochs/2023-01-17/19-43-26/checkpoints/epoch\=16/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_gaussian_unet_loss_no_weights_50epochs/2023-01-17/19-55-23/checkpoints/epoch\=27/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_gaussian_unet_loss_no_weights_50epochs/2023-01-17/20-07-27/checkpoints/epoch\=10/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_gaussian_unet_loss_no_weights_50epochs/2023-01-17/20-19-27/checkpoints/epoch\=20/backbone.pth")

training=("training-20" "training-10" "training-5")

for t in ${training[*]}; do
  devices="[4,5,6,7]"
  if [ "${t}" == "training-10" ]; then
    devices="[6,7]"
  fi
  if [ "${t}" == "training-5" ]; then
    devices="[7]"
  fi

  if [[ "${t}" != "training-20" ]]; then
    for j in ${weights_10[*]}; do
      params="experiment=fine_tune_cb55_AB1_train_20_run_unet.yaml
          trainer.devices=${devices}
          datamodule.train_folder_name=${t}
          +model.backbone.path_to_weights=${j}
          name=fine_tune_gaussian_cb55_AB1_${t}_unet_loss_no_weights_10pt_100e
          logger.wandb.tags=[unet,AB1,${t},3-classes,fine-tune,100-epochs,no-weights,gaussian,binary,10-epoch-pt]
          logger.wandb.group=fine-tune-gaussian-10pt-${t}"
      python run.py ${params}
    done

    for j in ${weights_20[*]}; do
      params="experiment=fine_tune_cb55_AB1_train_20_run_unet.yaml
          trainer.devices=${devices}
          datamodule.train_folder_name=${t}
          +model.backbone.path_to_weights=${j}
          name=fine_tune_gaussian_cb55_AB1_${t}_unet_loss_no_weights_20pt_100e
          logger.wandb.tags=[unet,AB1,${t},3-classes,fine-tune,100-epochs,no-weights,gaussian,binary,20-epoch-pt]
          logger.wandb.group=fine-tune-gaussian-20pt-${t}"
      python run.py ${params}
    done
  fi
  for j in ${weights_30[*]}; do
    params="experiment=fine_tune_cb55_AB1_train_20_run_unet.yaml
          trainer.devices=${devices}
          datamodule.train_folder_name=${t}
          +model.backbone.path_to_weights=${j}
          name=fine_tune_gaussian_cb55_AB1_${t}_unet_loss_no_weights_30pt_100e
          logger.wandb.tags=[unet,AB1,${t},3-classes,fine-tune,100-epochs,no-weights,gaussian,binary,30-epoch-pt]
          logger.wandb.group=fine-tune-gaussian-30pt-${t}"
    python run.py ${params}
  done

  for j in ${weights_40[*]}; do
    params="experiment=fine_tune_cb55_AB1_train_20_run_unet.yaml
          trainer.devices=${devices}
          datamodule.train_folder_name=${t}
          +model.backbone.path_to_weights=${j}
          name=fine_tune_gaussian_cb55_AB1_${t}_unet_loss_no_weights_40pt_100e
          logger.wandb.tags=[unet,AB1,${t},3-classes,fine-tune,100-epochs,no-weights,gaussian,binary,40-epoch-pt]
          logger.wandb.group=fine-tune-gaussian-40pt-${t}"
    python run.py ${params}
  done

  for j in ${weights_50[*]}; do
    params="experiment=fine_tune_cb55_AB1_train_20_run_unet.yaml
          trainer.devices=${devices}
          datamodule.train_folder_name=${t}
          +model.backbone.path_to_weights=${j}
          name=fine_tune_gaussian_cb55_AB1_${t}_unet_loss_no_weights_50pt_100e
          logger.wandb.tags=[unet,AB1,${t},3-classes,fine-tune,100-epochs,no-weights,gaussian,binary,50-epoch-pt]
          logger.wandb.group=fine-tune-gaussian-50pt-${t}"
    python run.py ${params}
  done
done
