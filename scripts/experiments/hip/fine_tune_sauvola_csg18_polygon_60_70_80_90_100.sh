#!/usr/bin/env bash

set -e

weights_60=("/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg18_sauvola_unet_loss_no_weights_60epoch/2023-03-22/15-10-56/checkpoints/epoch\=51/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg18_sauvola_unet_loss_no_weights_60epoch/2023-03-22/15-27-27/checkpoints/epoch\=56/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg18_sauvola_unet_loss_no_weights_60epoch/2023-03-22/15-43-55/checkpoints/epoch\=26/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg18_sauvola_unet_loss_no_weights_60epoch/2023-03-22/16-00-29/checkpoints/epoch\=45/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg18_sauvola_unet_loss_no_weights_60epoch/2023-03-22/16-17-07/checkpoints/epoch\=34/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg18_sauvola_unet_loss_no_weights_60epoch/2023-03-22/16-34-01/checkpoints/epoch\=37/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg18_sauvola_unet_loss_no_weights_60epoch/2023-03-22/16-50-36/checkpoints/epoch\=33/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg18_sauvola_unet_loss_no_weights_60epoch/2023-03-22/17-07-13/checkpoints/epoch\=23/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg18_sauvola_unet_loss_no_weights_60epoch/2023-03-22/17-23-42/checkpoints/epoch\=34/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg18_sauvola_unet_loss_no_weights_60epoch/2023-03-22/17-40-07/checkpoints/epoch\=33/backbone.pth"
)

devices="[0,1,2,3]"

for j in "${!weights_60[@]}"; do
  params="experiment=fine_tune_csg18_polygon_unet.yaml
          trainer.devices=${devices}
          mode=hip.yaml
          +model.backbone.path_to_weights=${weights_60[$j]}
          name=FT_sauvola_csg18_polygon_unet_loss_no_weights_60pt_100e
          logger.wandb.tags=[unet,csg18,polygon,3-classes,fine-tune,100-epochs,no-weights,sauvola,60-epoch-pt,with_header]
          logger.wandb.project=hip
          logger.wandb.group=fine-tune-sauvola-3cl-polygon-csg18-60pt"
  python run.py ${params}
  #    echo ${params}
done

weights_70=("/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg18_sauvola_unet_loss_no_weights_70epoch/2023-03-22/17-56-33/checkpoints/epoch\=54/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg18_sauvola_unet_loss_no_weights_70epoch/2023-03-22/18-15-28/checkpoints/epoch\=47/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg18_sauvola_unet_loss_no_weights_70epoch/2023-03-22/18-34-35/checkpoints/epoch\=46/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg18_sauvola_unet_loss_no_weights_70epoch/2023-03-22/18-53-39/checkpoints/epoch\=43/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg18_sauvola_unet_loss_no_weights_70epoch/2023-03-22/19-12-49/checkpoints/epoch\=43/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg18_sauvola_unet_loss_no_weights_70epoch/2023-03-22/19-31-50/checkpoints/epoch\=28/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg18_sauvola_unet_loss_no_weights_70epoch/2023-03-22/19-50-46/checkpoints/epoch\=27/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg18_sauvola_unet_loss_no_weights_70epoch/2023-03-22/20-09-47/checkpoints/epoch\=66/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg18_sauvola_unet_loss_no_weights_70epoch/2023-03-22/20-28-50/checkpoints/epoch\=26/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg18_sauvola_unet_loss_no_weights_70epoch/2023-03-22/20-47-59/checkpoints/epoch\=40/backbone.pth"
)

for j in "${!weights_70[@]}"; do
  params="experiment=fine_tune_csg18_polygon_unet.yaml
          trainer.devices=${devices}
          mode=hip.yaml
          +model.backbone.path_to_weights=${weights_70[$j]}
          name=FT_sauvola_csg18_polygon_unet_loss_no_weights_70pt_100e
          logger.wandb.tags=[unet,csg18,polygon,3-classes,fine-tune,100-epochs,no-weights,sauvola,70-epoch-pt,with_header]
          logger.wandb.project=hip
          logger.wandb.group=fine-tune-sauvola-3cl-polygon-csg18-70pt"
  python run.py ${params}
  #    echo ${params}
done

weights_80=("/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg18_sauvola_unet_loss_no_weights_80epoch/2023-03-22/21-06-53/checkpoints/epoch\=32/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg18_sauvola_unet_loss_no_weights_80epoch/2023-03-22/21-28-10/checkpoints/epoch\=50/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg18_sauvola_unet_loss_no_weights_80epoch/2023-03-22/21-49-31/checkpoints/epoch\=61/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg18_sauvola_unet_loss_no_weights_80epoch/2023-03-22/22-11-04/checkpoints/epoch\=56/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg18_sauvola_unet_loss_no_weights_80epoch/2023-03-22/22-32-37/checkpoints/epoch\=72/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg18_sauvola_unet_loss_no_weights_80epoch/2023-03-22/22-54-12/checkpoints/epoch\=73/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg18_sauvola_unet_loss_no_weights_80epoch/2023-03-22/23-15-54/checkpoints/epoch\=71/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg18_sauvola_unet_loss_no_weights_80epoch/2023-03-22/23-37-22/checkpoints/epoch\=21/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg18_sauvola_unet_loss_no_weights_80epoch/2023-03-22/23-58-47/checkpoints/epoch\=27/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg18_sauvola_unet_loss_no_weights_80epoch/2023-03-23/00-20-40/checkpoints/epoch\=39/backbone.pth"
)

for j in "${!weights_80[@]}"; do
  params="experiment=fine_tune_csg18_polygon_unet.yaml
          trainer.devices=${devices}
          mode=hip.yaml
          +model.backbone.path_to_weights=${weights_80[$j]}
          name=FT_sauvola_csg18_polygon_unet_loss_no_weights_80pt_100e
          logger.wandb.tags=[unet,csg18,polygon,3-classes,fine-tune,100-epochs,no-weights,sauvola,80-epoch-pt,with_header]
          logger.wandb.project=hip
          logger.wandb.group=fine-tune-sauvola-3cl-polygon-csg18-80pt"
  python run.py ${params}
  #    echo ${params}
done

weights_90=("/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg18_sauvola_unet_loss_no_weights_90epoch/2023-03-23/00-42-05/checkpoints/epoch\=20/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg18_sauvola_unet_loss_no_weights_90epoch/2023-03-23/01-06-01/checkpoints/epoch\=42/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg18_sauvola_unet_loss_no_weights_90epoch/2023-03-23/01-30-06/checkpoints/epoch\=37/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg18_sauvola_unet_loss_no_weights_90epoch/2023-03-23/01-54-07/checkpoints/epoch\=35/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg18_sauvola_unet_loss_no_weights_90epoch/2023-03-23/02-18-27/checkpoints/epoch\=60/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg18_sauvola_unet_loss_no_weights_90epoch/2023-03-23/02-42-29/checkpoints/epoch\=54/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg18_sauvola_unet_loss_no_weights_90epoch/2023-03-23/03-06-56/checkpoints/epoch\=49/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg18_sauvola_unet_loss_no_weights_90epoch/2023-03-23/03-31-24/checkpoints/epoch\=45/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg18_sauvola_unet_loss_no_weights_90epoch/2023-03-23/03-55-22/checkpoints/epoch\=36/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg18_sauvola_unet_loss_no_weights_90epoch/2023-03-23/04-19-23/checkpoints/epoch\=46/backbone.pth"
)

for j in "${!weights_90[@]}"; do
  params="experiment=fine_tune_csg18_polygon_unet.yaml
          trainer.devices=${devices}
          mode=hip.yaml
          +model.backbone.path_to_weights=${weights_90[$j]}
          name=FT_sauvola_csg18_polygon_unet_loss_no_weights_90pt_100e
          logger.wandb.tags=[unet,csg18,polygon,3-classes,fine-tune,100-epochs,no-weights,sauvola,90-epoch-pt,with_header]
          logger.wandb.project=hip
          logger.wandb.group=fine-tune-sauvola-3cl-polygon-csg18-90pt"
  python run.py ${params}
  #    echo ${params}
done

weights_100=("/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg18_sauvola_unet_loss_no_weights_100epoch/2023-03-23/04-43-17/checkpoints/epoch\=44/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg18_sauvola_unet_loss_no_weights_100epoch/2023-03-23/05-10-29/checkpoints/epoch\=57/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg18_sauvola_unet_loss_no_weights_100epoch/2023-03-23/05-37-08/checkpoints/epoch\=43/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg18_sauvola_unet_loss_no_weights_100epoch/2023-03-23/06-03-40/checkpoints/epoch\=30/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg18_sauvola_unet_loss_no_weights_100epoch/2023-03-23/06-30-27/checkpoints/epoch\=20/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg18_sauvola_unet_loss_no_weights_100epoch/2023-03-23/06-56-47/checkpoints/epoch\=64/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg18_sauvola_unet_loss_no_weights_100epoch/2023-03-23/07-23-38/checkpoints/epoch\=55/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg18_sauvola_unet_loss_no_weights_100epoch/2023-03-23/07-50-01/checkpoints/epoch\=82/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg18_sauvola_unet_loss_no_weights_100epoch/2023-03-23/08-16-19/checkpoints/epoch\=39/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg18_sauvola_unet_loss_no_weights_100epoch/2023-03-23/08-42-42/checkpoints/epoch\=75/backbone.pth"
)

for j in "${!weights_100[@]}"; do
  params="experiment=fine_tune_csg18_polygon_unet.yaml
          trainer.devices=${devices}
          mode=hip.yaml
          +model.backbone.path_to_weights=${weights_100[$j]}
          name=FT_sauvola_csg18_polygon_unet_loss_no_weights_100pt_100e
          logger.wandb.tags=[unet,csg18,polygon,3-classes,fine-tune,100-epochs,no-weights,sauvola,100-epoch-pt,with_header]
          logger.wandb.project=hip
          logger.wandb.group=fine-tune-sauvola-3cl-polygon-csg18-100pt"
  python run.py ${params}
  #    echo ${params}
done
