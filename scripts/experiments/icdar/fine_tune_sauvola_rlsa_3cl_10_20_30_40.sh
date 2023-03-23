#!/usr/bin/env bash

set -e

weights_10=("/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/3cl_rlsa_cb55_sauvola_unet_loss_no_weights_10epoch/2023-01-23/20-22-10/checkpoints/epoch\=9/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/3cl_rlsa_cb55_sauvola_unet_loss_no_weights_10epoch/2023-01-23/20-26-17/checkpoints/epoch\=3/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/3cl_rlsa_cb55_sauvola_unet_loss_no_weights_10epoch/2023-01-23/20-30-28/checkpoints/epoch\=4/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/3cl_rlsa_cb55_sauvola_unet_loss_no_weights_10epoch/2023-01-23/20-34-42/checkpoints/epoch\=7/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/3cl_rlsa_cb55_sauvola_unet_loss_no_weights_10epoch/2023-01-23/20-38-55/checkpoints/epoch\=6/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/3cl_rlsa_cb55_sauvola_unet_loss_no_weights_10epoch/2023-01-23/20-43-14/checkpoints/epoch\=2/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/3cl_rlsa_cb55_sauvola_unet_loss_no_weights_10epoch/2023-01-23/20-47-33/checkpoints/epoch\=9/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/3cl_rlsa_cb55_sauvola_unet_loss_no_weights_10epoch/2023-01-23/20-51-54/checkpoints/epoch\=5/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/3cl_rlsa_cb55_sauvola_unet_loss_no_weights_10epoch/2023-01-23/20-56-11/checkpoints/epoch\=6/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/3cl_rlsa_cb55_sauvola_unet_loss_no_weights_10epoch/2023-01-23/21-00-27/checkpoints/epoch\=4/backbone.pth")
weights_20=("/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/3cl_rlsa_cb55_sauvola_unet_loss_no_weights_20epoch/2023-01-23/21-04-41/checkpoints/epoch\=4/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/3cl_rlsa_cb55_sauvola_unet_loss_no_weights_20epoch/2023-01-23/21-11-54/checkpoints/epoch\=6/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/3cl_rlsa_cb55_sauvola_unet_loss_no_weights_20epoch/2023-01-23/21-19-09/checkpoints/epoch\=5/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/3cl_rlsa_cb55_sauvola_unet_loss_no_weights_20epoch/2023-01-23/21-26-28/checkpoints/epoch\=5/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/3cl_rlsa_cb55_sauvola_unet_loss_no_weights_20epoch/2023-01-23/21-33-49/checkpoints/epoch\=4/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/3cl_rlsa_cb55_sauvola_unet_loss_no_weights_20epoch/2023-01-23/21-41-11/checkpoints/epoch\=10/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/3cl_rlsa_cb55_sauvola_unet_loss_no_weights_20epoch/2023-01-23/21-48-34/checkpoints/epoch\=14/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/3cl_rlsa_cb55_sauvola_unet_loss_no_weights_20epoch/2023-01-23/21-56-06/checkpoints/epoch\=12/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/3cl_rlsa_cb55_sauvola_unet_loss_no_weights_20epoch/2023-01-23/22-03-27/checkpoints/epoch\=2/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/3cl_rlsa_cb55_sauvola_unet_loss_no_weights_20epoch/2023-01-23/22-10-43/checkpoints/epoch\=9/backbone.pth")
weights_30=("/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/3cl_rlsa_cb55_sauvola_unet_loss_no_weights_30epoch/2023-01-23/22-18-18/checkpoints/epoch\=13/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/3cl_rlsa_cb55_sauvola_unet_loss_no_weights_30epoch/2023-01-23/22-28-48/checkpoints/epoch\=6/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/3cl_rlsa_cb55_sauvola_unet_loss_no_weights_30epoch/2023-01-23/22-39-23/checkpoints/epoch\=23/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/3cl_rlsa_cb55_sauvola_unet_loss_no_weights_30epoch/2023-01-23/22-49-59/checkpoints/epoch\=19/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/3cl_rlsa_cb55_sauvola_unet_loss_no_weights_30epoch/2023-01-23/23-00-32/checkpoints/epoch\=6/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/3cl_rlsa_cb55_sauvola_unet_loss_no_weights_30epoch/2023-01-23/23-10-58/checkpoints/epoch\=14/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/3cl_rlsa_cb55_sauvola_unet_loss_no_weights_30epoch/2023-01-23/23-21-17/checkpoints/epoch\=23/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/3cl_rlsa_cb55_sauvola_unet_loss_no_weights_30epoch/2023-01-23/23-31-30/checkpoints/epoch\=17/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/3cl_rlsa_cb55_sauvola_unet_loss_no_weights_30epoch/2023-01-23/23-41-47/checkpoints/epoch\=3/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/3cl_rlsa_cb55_sauvola_unet_loss_no_weights_30epoch/2023-01-23/23-52-02/checkpoints/epoch\=7/backbone.pth")
weights_40=("/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/3cl_rlsa_cb55_sauvola_unet_loss_no_weights_40epoch/2023-01-24/00-02-14/checkpoints/epoch\=15/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/3cl_rlsa_cb55_sauvola_unet_loss_no_weights_40epoch/2023-01-24/00-15-27/checkpoints/epoch\=19/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/3cl_rlsa_cb55_sauvola_unet_loss_no_weights_40epoch/2023-01-24/00-28-45/checkpoints/epoch\=35/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/3cl_rlsa_cb55_sauvola_unet_loss_no_weights_40epoch/2023-01-24/00-42-07/checkpoints/epoch\=17/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/3cl_rlsa_cb55_sauvola_unet_loss_no_weights_40epoch/2023-01-24/00-55-18/checkpoints/epoch\=5/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/3cl_rlsa_cb55_sauvola_unet_loss_no_weights_40epoch/2023-01-24/01-08-28/checkpoints/epoch\=32/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/3cl_rlsa_cb55_sauvola_unet_loss_no_weights_40epoch/2023-01-24/01-21-45/checkpoints/epoch\=16/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/3cl_rlsa_cb55_sauvola_unet_loss_no_weights_40epoch/2023-01-24/01-34-59/checkpoints/epoch\=13/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/3cl_rlsa_cb55_sauvola_unet_loss_no_weights_40epoch/2023-01-24/01-48-11/checkpoints/epoch\=5/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/3cl_rlsa_cb55_sauvola_unet_loss_no_weights_40epoch/2023-01-24/02-01-31/checkpoints/epoch\=38/backbone.pth")

training=("training-10" "training-5")

for t in ${training[*]}; do
  devices="[4,5,6,7]"
  if [ "${t}" == "training-10" ]; then
    devices="[6,7]"
  fi
  if [ "${t}" == "training-5" ]; then
    devices="[7]"
  fi

  if [[ "${t}" != "training-10" ]]; then
    for j in ${weights_10[*]}; do
      params="experiment=fine_tune_cb55_AB1_train_20_run_unet.yaml
          trainer.devices=${devices}
          datamodule.train_folder_name=${t}
          +model.backbone.path_to_weights=${j}
          name=fine_tune_sauvola_rlsa_3cl_cb55_AB1_${t}_unet_loss_no_weights_10pt_100e
          logger.wandb.tags=[unet,AB1,${t},3-classes,fine-tune,100-epochs,no-weights,3cl,rlsa_3cl,10-epoch-pt]
          logger.wandb.group=fine-tune-sauvola-rlsa-3cl-10pt-${t}"
      python run.py ${params}
    done

    for j in ${weights_20[*]}; do
      params="experiment=fine_tune_cb55_AB1_train_20_run_unet.yaml
          trainer.devices=${devices}
          datamodule.train_folder_name=${t}
          +model.backbone.path_to_weights=${j}
          name=fine_tune_sauvola_rlsa_3cl_cb55_AB1_${t}_unet_loss_no_weights_20pt_100e
          logger.wandb.tags=[unet,AB1,${t},3-classes,fine-tune,100-epochs,no-weights,3cl,rlsa_3cl,20-epoch-pt]
          logger.wandb.group=fine-tune-sauvola-rlsa-3cl-20pt-${t}"
      python run.py ${params}
    done

    for j in ${weights_30[*]}; do
      params="experiment=fine_tune_cb55_AB1_train_20_run_unet.yaml
          trainer.devices=${devices}
          datamodule.train_folder_name=${t}
          +model.backbone.path_to_weights=${j}
          name=fine_tune_sauvola_rlsa_3cl_cb55_AB1_${t}_unet_loss_no_weights_30pt_100e
          logger.wandb.tags=[unet,AB1,${t},3-classes,fine-tune,100-epochs,no-weights,3cl,rlsa_3cl,30-epoch-pt]
          logger.wandb.group=fine-tune-sauvola-rlsa-3cl-30pt-${t}"
      python run.py ${params}
    done

  fi

  for j in ${weights_40[*]}; do
    params="experiment=fine_tune_cb55_AB1_train_20_run_unet.yaml
          trainer.devices=${devices}
          datamodule.train_folder_name=${t}
          +model.backbone.path_to_weights=${j}
          name=fine_tune_sauvola_rlsa_3cl_cb55_AB1_${t}_unet_loss_no_weights_40pt_100e
          logger.wandb.tags=[unet,AB1,${t},3-classes,fine-tune,100-epochs,no-weights,3cl,rlsa_3cl,40-epoch-pt]
          logger.wandb.group=fine-tune-sauvola-rlsa-3cl-40pt-${t}"
    python run.py ${params}
  done
done
