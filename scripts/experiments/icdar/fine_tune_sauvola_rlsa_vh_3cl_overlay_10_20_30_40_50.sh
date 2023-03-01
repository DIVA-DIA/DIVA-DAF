#!/usr/bin/env bash

set -e

weights_10=("/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/3cl_rlsa_vh_overlay_cb55_sauvola_unet_loss_no_weights_10epoch/2023-01-26/18-57-12/checkpoints/epoch\=4/backbone.pth"
"/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/3cl_rlsa_vh_overlay_cb55_sauvola_unet_loss_no_weights_10epoch/2023-01-26/19-00-23/checkpoints/epoch\=9/backbone.pth"
"/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/3cl_rlsa_vh_overlay_cb55_sauvola_unet_loss_no_weights_10epoch/2023-01-26/19-03-29/checkpoints/epoch\=2/backbone.pth"
"/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/3cl_rlsa_vh_overlay_cb55_sauvola_unet_loss_no_weights_10epoch/2023-01-26/19-06-32/checkpoints/epoch\=4/backbone.pth"
"/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/3cl_rlsa_vh_overlay_cb55_sauvola_unet_loss_no_weights_10epoch/2023-01-26/19-09-36/checkpoints/epoch\=5/backbone.pth"
"/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/3cl_rlsa_vh_overlay_cb55_sauvola_unet_loss_no_weights_10epoch/2023-01-26/19-12-38/checkpoints/epoch\=5/backbone.pth"
"/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/3cl_rlsa_vh_overlay_cb55_sauvola_unet_loss_no_weights_10epoch/2023-01-26/19-15-41/checkpoints/epoch\=2/backbone.pth"
"/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/3cl_rlsa_vh_overlay_cb55_sauvola_unet_loss_no_weights_10epoch/2023-01-26/19-18-38/checkpoints/epoch\=4/backbone.pth"
"/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/3cl_rlsa_vh_overlay_cb55_sauvola_unet_loss_no_weights_10epoch/2023-01-26/19-21-38/checkpoints/epoch\=6/backbone.pth"
"/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/3cl_rlsa_vh_overlay_cb55_sauvola_unet_loss_no_weights_10epoch/2023-01-26/19-24-35/checkpoints/epoch\=4/backbone.pth")
weights_20=("/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/3cl_rlsa_vh_overlay_cb55_sauvola_unet_loss_no_weights_20epoch/2023-01-26/19-27-35/checkpoints/epoch\=17/backbone.pth"
"/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/3cl_rlsa_vh_overlay_cb55_sauvola_unet_loss_no_weights_20epoch/2023-01-26/19-32-42/checkpoints/epoch\=3/backbone.pth"
"/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/3cl_rlsa_vh_overlay_cb55_sauvola_unet_loss_no_weights_20epoch/2023-01-26/19-37-45/checkpoints/epoch\=11/backbone.pth"
"/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/3cl_rlsa_vh_overlay_cb55_sauvola_unet_loss_no_weights_20epoch/2023-01-26/19-42-55/checkpoints/epoch\=9/backbone.pth"
"/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/3cl_rlsa_vh_overlay_cb55_sauvola_unet_loss_no_weights_20epoch/2023-01-26/19-48-15/checkpoints/epoch\=11/backbone.pth"
"/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/3cl_rlsa_vh_overlay_cb55_sauvola_unet_loss_no_weights_20epoch/2023-01-26/19-53-19/checkpoints/epoch\=12/backbone.pth"
"/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/3cl_rlsa_vh_overlay_cb55_sauvola_unet_loss_no_weights_20epoch/2023-01-26/19-58-33/checkpoints/epoch\=17/backbone.pth"
"/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/3cl_rlsa_vh_overlay_cb55_sauvola_unet_loss_no_weights_20epoch/2023-01-26/20-03-42/checkpoints/epoch\=12/backbone.pth"
"/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/3cl_rlsa_vh_overlay_cb55_sauvola_unet_loss_no_weights_20epoch/2023-01-26/20-08-44/checkpoints/epoch\=18/backbone.pth"
"/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/3cl_rlsa_vh_overlay_cb55_sauvola_unet_loss_no_weights_20epoch/2023-01-26/20-13-53/checkpoints/epoch\=5/backbone.pth")
weights_30=("/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/3cl_rlsa_vh_overlay_cb55_sauvola_unet_loss_no_weights_30epoch/2023-01-26/20-18-58/checkpoints/epoch\=12/backbone.pth"
"/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/3cl_rlsa_vh_overlay_cb55_sauvola_unet_loss_no_weights_30epoch/2023-01-26/20-26-12/checkpoints/epoch\=10/backbone.pth"
"/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/3cl_rlsa_vh_overlay_cb55_sauvola_unet_loss_no_weights_30epoch/2023-01-26/20-33-23/checkpoints/epoch\=6/backbone.pth"
"/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/3cl_rlsa_vh_overlay_cb55_sauvola_unet_loss_no_weights_30epoch/2023-01-26/20-40-38/checkpoints/epoch\=17/backbone.pth"
"/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/3cl_rlsa_vh_overlay_cb55_sauvola_unet_loss_no_weights_30epoch/2023-01-26/20-47-55/checkpoints/epoch\=24/backbone.pth"
"/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/3cl_rlsa_vh_overlay_cb55_sauvola_unet_loss_no_weights_30epoch/2023-01-26/20-55-06/checkpoints/epoch\=13/backbone.pth"
"/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/3cl_rlsa_vh_overlay_cb55_sauvola_unet_loss_no_weights_30epoch/2023-01-26/21-02-22/checkpoints/epoch\=18/backbone.pth"
"/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/3cl_rlsa_vh_overlay_cb55_sauvola_unet_loss_no_weights_30epoch/2023-01-26/21-09-37/checkpoints/epoch\=14/backbone.pth"
"/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/3cl_rlsa_vh_overlay_cb55_sauvola_unet_loss_no_weights_30epoch/2023-01-26/21-16-56/checkpoints/epoch\=19/backbone.pth"
"/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/3cl_rlsa_vh_overlay_cb55_sauvola_unet_loss_no_weights_30epoch/2023-01-26/21-24-19/checkpoints/epoch\=5/backbone.pth")
weights_40=("/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/3cl_rlsa_vh_overlay_cb55_sauvola_unet_loss_no_weights_40epoch/2023-01-26/21-31-38/checkpoints/epoch\=33/backbone.pth"
"/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/3cl_rlsa_vh_overlay_cb55_sauvola_unet_loss_no_weights_40epoch/2023-01-26/21-41-15/checkpoints/epoch\=2/backbone.pth"
"/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/3cl_rlsa_vh_overlay_cb55_sauvola_unet_loss_no_weights_40epoch/2023-01-26/21-50-43/checkpoints/epoch\=14/backbone.pth"
"/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/3cl_rlsa_vh_overlay_cb55_sauvola_unet_loss_no_weights_40epoch/2023-01-26/22-00-28/checkpoints/epoch\=24/backbone.pth"
"/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/3cl_rlsa_vh_overlay_cb55_sauvola_unet_loss_no_weights_40epoch/2023-01-26/22-09-56/checkpoints/epoch\=23/backbone.pth"
"/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/3cl_rlsa_vh_overlay_cb55_sauvola_unet_loss_no_weights_40epoch/2023-01-26/22-19-20/checkpoints/epoch\=39/backbone.pth"
"/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/3cl_rlsa_vh_overlay_cb55_sauvola_unet_loss_no_weights_40epoch/2023-01-26/22-28-45/checkpoints/epoch\=18/backbone.pth"
"/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/3cl_rlsa_vh_overlay_cb55_sauvola_unet_loss_no_weights_40epoch/2023-01-26/22-38-11/checkpoints/epoch\=27/backbone.pth"
"/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/3cl_rlsa_vh_overlay_cb55_sauvola_unet_loss_no_weights_40epoch/2023-01-26/22-47-42/checkpoints/epoch\=15/backbone.pth"
"/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/3cl_rlsa_vh_overlay_cb55_sauvola_unet_loss_no_weights_40epoch/2023-01-26/22-57-11/checkpoints/epoch\=32/backbone.pth")
weights_50=("/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/3cl_rlsa_vh_overlay_cb55_sauvola_unet_loss_no_weights_50epoch/2023-01-26/23-06-38/checkpoints/epoch\=49/backbone.pth"
"/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/3cl_rlsa_vh_overlay_cb55_sauvola_unet_loss_no_weights_50epoch/2023-01-26/23-18-14/checkpoints/epoch\=13/backbone.pth"
"/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/3cl_rlsa_vh_overlay_cb55_sauvola_unet_loss_no_weights_50epoch/2023-01-26/23-29-45/checkpoints/epoch\=13/backbone.pth"
"/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/3cl_rlsa_vh_overlay_cb55_sauvola_unet_loss_no_weights_50epoch/2023-01-26/23-41-17/checkpoints/epoch\=22/backbone.pth"
"/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/3cl_rlsa_vh_overlay_cb55_sauvola_unet_loss_no_weights_50epoch/2023-01-26/23-52-47/checkpoints/epoch\=46/backbone.pth"
"/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/3cl_rlsa_vh_overlay_cb55_sauvola_unet_loss_no_weights_50epoch/2023-01-27/00-04-23/checkpoints/epoch\=10/backbone.pth"
"/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/3cl_rlsa_vh_overlay_cb55_sauvola_unet_loss_no_weights_50epoch/2023-01-27/00-15-56/checkpoints/epoch\=22/backbone.pth"
"/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/3cl_rlsa_vh_overlay_cb55_sauvola_unet_loss_no_weights_50epoch/2023-01-27/00-27-26/checkpoints/epoch\=13/backbone.pth"
"/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/3cl_rlsa_vh_overlay_cb55_sauvola_unet_loss_no_weights_50epoch/2023-01-27/00-39-00/checkpoints/epoch\=15/backbone.pth"
"/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/3cl_rlsa_vh_overlay_cb55_sauvola_unet_loss_no_weights_50epoch/2023-01-27/00-50-29/checkpoints/epoch\=26/backbone.pth")

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
          name=fine_tune_sauvola_rlsa_vh_7cl_overlay_cb55_AB1_${t}_unet_loss_no_weights_10pt_100e
          logger.wandb.tags=[unet,AB1,${t},3-classes,fine-tune,100-epochs,no-weights,7cl,rlsa_vh_3cl_overlay,10-epoch-pt]
          logger.wandb.group=fine-tune-sauvola-rlsa-7cl-overlay-10pt-${t}"
    python run.py ${params}
  done

  for j in ${weights_20[*]}; do
    params="experiment=fine_tune_cb55_AB1_train_20_run_unet.yaml
          trainer.devices=${devices}
          datamodule.train_folder_name=${t}
          +model.backbone.path_to_weights=${j}
          name=fine_tune_sauvola_rlsa_vh_7cl_overlay_cb55_AB1_${t}_unet_loss_no_weights_20pt_100e
          logger.wandb.tags=[unet,AB1,${t},3-classes,fine-tune,100-epochs,no-weights,7cl,rlsa_vh_3cl_overlay,20-epoch-pt]
          logger.wandb.group=fine-tune-sauvola-rlsa-7cl-overlay-20pt-${t}"
    python run.py ${params}
  done

  for j in ${weights_30[*]}; do
    params="experiment=fine_tune_cb55_AB1_train_20_run_unet.yaml
          trainer.devices=${devices}
          datamodule.train_folder_name=${t}
          +model.backbone.path_to_weights=${j}
          name=fine_tune_sauvola_rlsa_vh_7cl_overlay_cb55_AB1_${t}_unet_loss_no_weights_30pt_100e
          logger.wandb.tags=[unet,AB1,${t},3-classes,fine-tune,100-epochs,no-weights,7cl,rlsa_vh_3cl_overlay,30-epoch-pt]
          logger.wandb.group=fine-tune-sauvola-rlsa-7cl-overlay-30pt-${t}"
    python run.py ${params}
  done

  for j in ${weights_40[*]}; do
    params="experiment=fine_tune_cb55_AB1_train_20_run_unet.yaml
          trainer.devices=${devices}
          datamodule.train_folder_name=${t}
          +model.backbone.path_to_weights=${j}
          name=fine_tune_sauvola_rlsa_vh_7cl_overlay_cb55_AB1_${t}_unet_loss_no_weights_40pt_100e
          logger.wandb.tags=[unet,AB1,${t},3-classes,fine-tune,100-epochs,no-weights,7cl,rlsa_vh_3cl_overlay,40-epoch-pt]
          logger.wandb.group=fine-tune-sauvola-rlsa-7cl-overlay-40pt-${t}"
    python run.py ${params}
  done

  for j in ${weights_40[*]}; do
    params="experiment=fine_tune_cb55_AB1_train_20_run_unet.yaml
          trainer.devices=${devices}
          datamodule.train_folder_name=${t}
          +model.backbone.path_to_weights=${j}
          name=fine_tune_sauvola_rlsa_vh_7cl_overlay_cb55_AB1_${t}_unet_loss_no_weights_50pt_100e
          logger.wandb.tags=[unet,AB1,${t},3-classes,fine-tune,100-epochs,no-weights,7cl,rlsa_vh_3cl_overlay,50-epoch-pt]
          logger.wandb.group=fine-tune-sauvola-rlsa-7cl-overlay-50pt-${t}"
    python run.py ${params}
  done
done
