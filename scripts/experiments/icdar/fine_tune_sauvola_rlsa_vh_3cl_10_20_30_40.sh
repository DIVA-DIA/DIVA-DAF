#!/usr/bin/env bash

set -e

weights_10=("/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/3cl_rlsa_vh_cb55_sauvola_unet_loss_no_weights_10epoch/2023-01-25/15-43-48/checkpoints/epoch\=9/backbone.pth"
"/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/3cl_rlsa_vh_cb55_sauvola_unet_loss_no_weights_10epoch/2023-01-25/15-46-49/checkpoints/epoch\=3/backbone.pth"
"/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/3cl_rlsa_vh_cb55_sauvola_unet_loss_no_weights_10epoch/2023-01-25/15-49-44/checkpoints/epoch\=9/backbone.pth"
"/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/3cl_rlsa_vh_cb55_sauvola_unet_loss_no_weights_10epoch/2023-01-25/15-52-42/checkpoints/epoch\=9/backbone.pth"
"/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/3cl_rlsa_vh_cb55_sauvola_unet_loss_no_weights_10epoch/2023-01-25/15-55-35/checkpoints/epoch\=4/backbone.pth"
"/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/3cl_rlsa_vh_cb55_sauvola_unet_loss_no_weights_10epoch/2023-01-25/15-58-33/checkpoints/epoch\=7/backbone.pth"
"/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/3cl_rlsa_vh_cb55_sauvola_unet_loss_no_weights_10epoch/2023-01-25/16-01-29/checkpoints/epoch\=7/backbone.pth"
"/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/3cl_rlsa_vh_cb55_sauvola_unet_loss_no_weights_10epoch/2023-01-25/16-04-22/checkpoints/epoch\=7/backbone.pth"
"/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/3cl_rlsa_vh_cb55_sauvola_unet_loss_no_weights_10epoch/2023-01-25/16-07-15/checkpoints/epoch\=1/backbone.pth"
"/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/3cl_rlsa_vh_cb55_sauvola_unet_loss_no_weights_10epoch/2023-01-25/16-10-07/checkpoints/epoch\=9/backbone.pth")
weights_20=("/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/3cl_rlsa_vh_cb55_sauvola_unet_loss_no_weights_20epoch/2023-01-25/16-13-04/checkpoints/epoch\=3/backbone.pth"
"/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/3cl_rlsa_vh_cb55_sauvola_unet_loss_no_weights_20epoch/2023-01-25/16-17-58/checkpoints/epoch\=3/backbone.pth"
"/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/3cl_rlsa_vh_cb55_sauvola_unet_loss_no_weights_20epoch/2023-01-25/16-22-49/checkpoints/epoch\=13/backbone.pth"
"/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/3cl_rlsa_vh_cb55_sauvola_unet_loss_no_weights_20epoch/2023-01-25/16-27-48/checkpoints/epoch\=9/backbone.pth"
"/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/3cl_rlsa_vh_cb55_sauvola_unet_loss_no_weights_20epoch/2023-01-25/16-32-40/checkpoints/epoch\=3/backbone.pth"
"/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/3cl_rlsa_vh_cb55_sauvola_unet_loss_no_weights_20epoch/2023-01-25/16-37-30/checkpoints/epoch\=8/backbone.pth"
"/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/3cl_rlsa_vh_cb55_sauvola_unet_loss_no_weights_20epoch/2023-01-25/16-42-20/checkpoints/epoch\=13/backbone.pth"
"/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/3cl_rlsa_vh_cb55_sauvola_unet_loss_no_weights_20epoch/2023-01-25/16-47-17/checkpoints/epoch\=10/backbone.pth"
"/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/3cl_rlsa_vh_cb55_sauvola_unet_loss_no_weights_20epoch/2023-01-25/16-52-11/checkpoints/epoch\=9/backbone.pth"
"/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/3cl_rlsa_vh_cb55_sauvola_unet_loss_no_weights_20epoch/2023-01-25/16-57-05/checkpoints/epoch\=14/backbone.pth")
weights_30=("/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/3cl_rlsa_vh_cb55_sauvola_unet_loss_no_weights_30epoch/2023-01-25/17-01-59/checkpoints/epoch\=24/backbone.pth"
"/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/3cl_rlsa_vh_cb55_sauvola_unet_loss_no_weights_30epoch/2023-01-25/17-08-48/checkpoints/epoch\=22/backbone.pth"
"/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/3cl_rlsa_vh_cb55_sauvola_unet_loss_no_weights_30epoch/2023-01-25/17-15-41/checkpoints/epoch\=28/backbone.pth"
"/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/3cl_rlsa_vh_cb55_sauvola_unet_loss_no_weights_30epoch/2023-01-25/17-22-33/checkpoints/epoch\=3/backbone.pth"
"/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/3cl_rlsa_vh_cb55_sauvola_unet_loss_no_weights_30epoch/2023-01-25/17-29-21/checkpoints/epoch\=24/backbone.pth"
"/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/3cl_rlsa_vh_cb55_sauvola_unet_loss_no_weights_30epoch/2023-01-25/17-36-16/checkpoints/epoch\=19/backbone.pth"
"/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/3cl_rlsa_vh_cb55_sauvola_unet_loss_no_weights_30epoch/2023-01-25/17-43-06/checkpoints/epoch\=20/backbone.pth"
"/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/3cl_rlsa_vh_cb55_sauvola_unet_loss_no_weights_30epoch/2023-01-25/17-49-56/checkpoints/epoch\=10/backbone.pth"
"/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/3cl_rlsa_vh_cb55_sauvola_unet_loss_no_weights_30epoch/2023-01-25/17-56-46/checkpoints/epoch\=6/backbone.pth"
"/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/3cl_rlsa_vh_cb55_sauvola_unet_loss_no_weights_30epoch/2023-01-25/18-03-36/checkpoints/epoch\=11/backbone.pth")
weights_40=("/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/3cl_rlsa_vh_cb55_sauvola_unet_loss_no_weights_40epoch/2023-01-25/18-10-29/checkpoints/epoch\=6/backbone.pth"
"/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/3cl_rlsa_vh_cb55_sauvola_unet_loss_no_weights_40epoch/2023-01-25/18-19-16/checkpoints/epoch\=31/backbone.pth"
"/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/3cl_rlsa_vh_cb55_sauvola_unet_loss_no_weights_40epoch/2023-01-25/18-28-12/checkpoints/epoch\=8/backbone.pth"
"/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/3cl_rlsa_vh_cb55_sauvola_unet_loss_no_weights_40epoch/2023-01-25/18-36-58/checkpoints/epoch\=11/backbone.pth"
"/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/3cl_rlsa_vh_cb55_sauvola_unet_loss_no_weights_40epoch/2023-01-25/18-45-44/checkpoints/epoch\=4/backbone.pth"
"/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/3cl_rlsa_vh_cb55_sauvola_unet_loss_no_weights_40epoch/2023-01-25/18-54-26/checkpoints/epoch\=25/backbone.pth"
"/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/3cl_rlsa_vh_cb55_sauvola_unet_loss_no_weights_40epoch/2023-01-25/19-03-20/checkpoints/epoch\=14/backbone.pth"
"/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/3cl_rlsa_vh_cb55_sauvola_unet_loss_no_weights_40epoch/2023-01-25/19-12-07/checkpoints/epoch\=18/backbone.pth"
"/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/3cl_rlsa_vh_cb55_sauvola_unet_loss_no_weights_40epoch/2023-01-25/19-21-02/checkpoints/epoch\=6/backbone.pth"
"/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/3cl_rlsa_vh_cb55_sauvola_unet_loss_no_weights_40epoch/2023-01-25/19-29-49/checkpoints/epoch\=12/backbone.pth")

training=("training-20")

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
          name=fine_tune_sauvola_rlsa_vh_3cl_cb55_AB1_${t}_unet_loss_no_weights_10pt_100e
          logger.wandb.tags=[unet,AB1,${t},3-classes,fine-tune,100-epochs,no-weights,3cl,rlsa_vh_3cl,10-epoch-pt]
          logger.wandb.group=fine-tune-sauvola-rlsa-3cl-10pt-${t}"
    python run.py ${params}
  done

  for j in ${weights_20[*]}; do
    params="experiment=fine_tune_cb55_AB1_train_20_run_unet.yaml
          trainer.devices=${devices}
          datamodule.train_folder_name=${t}
          +model.backbone.path_to_weights=${j}
          name=fine_tune_sauvola_rlsa_vh_3cl_cb55_AB1_${t}_unet_loss_no_weights_20pt_100e
          logger.wandb.tags=[unet,AB1,${t},3-classes,fine-tune,100-epochs,no-weights,3cl,rlsa_vh_3cl,20-epoch-pt]
          logger.wandb.group=fine-tune-sauvola-rlsa-3cl-20pt-${t}"
    python run.py ${params}
  done

  for j in ${weights_30[*]}; do
    params="experiment=fine_tune_cb55_AB1_train_20_run_unet.yaml
          trainer.devices=${devices}
          datamodule.train_folder_name=${t}
          +model.backbone.path_to_weights=${j}
          name=fine_tune_sauvola_rlsa_vh_3cl_cb55_AB1_${t}_unet_loss_no_weights_30pt_100e
          logger.wandb.tags=[unet,AB1,${t},3-classes,fine-tune,100-epochs,no-weights,3cl,rlsa_vh_3cl,30-epoch-pt]
          logger.wandb.group=fine-tune-sauvola-rlsa-3cl-30pt-${t}"
    python run.py ${params}
  done

  for j in ${weights_40[*]}; do
    params="experiment=fine_tune_cb55_AB1_train_20_run_unet.yaml
          trainer.devices=${devices}
          datamodule.train_folder_name=${t}
          +model.backbone.path_to_weights=${j}
          name=fine_tune_sauvola_rlsa_vh_3cl_cb55_AB1_${t}_unet_loss_no_weights_40pt_100e
          logger.wandb.tags=[unet,AB1,${t},3-classes,fine-tune,100-epochs,no-weights,3cl,rlsa_vh_3cl,40-epoch-pt]
          logger.wandb.group=fine-tune-sauvola-rlsa-3cl-40pt-${t}"
    python run.py ${params}
  done
done
