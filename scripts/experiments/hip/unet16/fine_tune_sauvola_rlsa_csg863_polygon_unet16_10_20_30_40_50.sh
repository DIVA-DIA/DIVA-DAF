#!/usr/bin/env bash

set -e

weights_10=("/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg863_sauvola_rlsa_unet16_loss_no_weights_10epoch/2023-04-21/02-58-48/checkpoints/epoch\=9/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg863_sauvola_rlsa_unet16_loss_no_weights_10epoch/2023-04-21/03-00-55/checkpoints/epoch\=8/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg863_sauvola_rlsa_unet16_loss_no_weights_10epoch/2023-04-21/03-03-00/checkpoints/epoch\=8/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg863_sauvola_rlsa_unet16_loss_no_weights_10epoch/2023-04-21/03-05-05/checkpoints/epoch\=9/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg863_sauvola_rlsa_unet16_loss_no_weights_10epoch/2023-04-21/03-07-07/checkpoints/epoch\=9/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg863_sauvola_rlsa_unet16_loss_no_weights_10epoch/2023-04-21/03-09-18/checkpoints/epoch\=9/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg863_sauvola_rlsa_unet16_loss_no_weights_10epoch/2023-04-21/03-11-24/checkpoints/epoch\=9/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg863_sauvola_rlsa_unet16_loss_no_weights_10epoch/2023-04-21/03-13-32/checkpoints/epoch\=9/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg863_sauvola_rlsa_unet16_loss_no_weights_10epoch/2023-04-21/03-15-36/checkpoints/epoch\=9/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg863_sauvola_rlsa_unet16_loss_no_weights_10epoch/2023-04-21/03-17-43/checkpoints/epoch\=9/backbone.pth"
)

devices="[0,1,2,3]"

for j in "${!weights_10[@]}"; do
  params="experiment=fine_tune_csg863_polygon_unet.yaml
          trainer.devices=${devices}
          mode=hip.yaml
          +model.backbone.path_to_weights=${weights_10[$j]}
          name=FT_sauvola_rlsa_csg863_polygon_unet16_loss_no_weights_10pt_100e
          logger.wandb.tags=[unet16,csg863,polygon,3-classes,fine-tune,100-epochs,no-weights,sauvola_rlsa,10-epoch-pt,with_header]
          logger.wandb.project=hip
          logger.wandb.group=fine-tune-sauvola_rlsa-3cl-polygon-csg863-10pt"
  python run.py ${params}
  #    echo ${params}
done

weights_20=("/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg863_sauvola_rlsa_unet16_loss_no_weights_20epoch/2023-04-21/03-19-45/checkpoints/epoch\=17/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg863_sauvola_rlsa_unet16_loss_no_weights_20epoch/2023-04-21/03-22-58/checkpoints/epoch\=19/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg863_sauvola_rlsa_unet16_loss_no_weights_20epoch/2023-04-21/03-26-13/checkpoints/epoch\=16/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg863_sauvola_rlsa_unet16_loss_no_weights_20epoch/2023-04-21/03-29-28/checkpoints/epoch\=17/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg863_sauvola_rlsa_unet16_loss_no_weights_20epoch/2023-04-21/03-32-46/checkpoints/epoch\=19/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg863_sauvola_rlsa_unet16_loss_no_weights_20epoch/2023-04-21/03-36-01/checkpoints/epoch\=18/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg863_sauvola_rlsa_unet16_loss_no_weights_20epoch/2023-04-21/03-39-16/checkpoints/epoch\=19/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg863_sauvola_rlsa_unet16_loss_no_weights_20epoch/2023-04-21/03-42-32/checkpoints/epoch\=18/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg863_sauvola_rlsa_unet16_loss_no_weights_20epoch/2023-04-21/03-45-49/checkpoints/epoch\=17/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg863_sauvola_rlsa_unet16_loss_no_weights_20epoch/2023-04-21/03-49-04/checkpoints/epoch\=19/backbone.pth"
)

for j in "${!weights_20[@]}"; do
  params="experiment=fine_tune_csg863_polygon_unet.yaml
          trainer.devices=${devices}
          mode=hip.yaml
          +model.backbone.path_to_weights=${weights_20[$j]}
          name=FT_sauvola_rlsa_csg863_polygon_unet16_loss_no_weights_20pt_100e
          logger.wandb.tags=[unet16,csg863,polygon,3-classes,fine-tune,100-epochs,no-weights,sauvola_rlsa,20-epoch-pt,with_header]
          logger.wandb.project=hip
          logger.wandb.group=fine-tune-sauvola_rlsa-3cl-polygon-csg863-20pt"
  python run.py ${params}
  #    echo ${params}
done

weights_30=("/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg863_sauvola_rlsa_unet16_loss_no_weights_30epoch/2023-04-21/03-52-23/checkpoints/epoch\=22/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg863_sauvola_rlsa_unet16_loss_no_weights_30epoch/2023-04-21/03-56-53/checkpoints/epoch\=23/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg863_sauvola_rlsa_unet16_loss_no_weights_30epoch/2023-04-21/04-01-21/checkpoints/epoch\=29/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg863_sauvola_rlsa_unet16_loss_no_weights_30epoch/2023-04-21/04-05-50/checkpoints/epoch\=29/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg863_sauvola_rlsa_unet16_loss_no_weights_30epoch/2023-04-21/04-10-16/checkpoints/epoch\=29/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg863_sauvola_rlsa_unet16_loss_no_weights_30epoch/2023-04-21/04-14-41/checkpoints/epoch\=25/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg863_sauvola_rlsa_unet16_loss_no_weights_30epoch/2023-04-21/04-19-07/checkpoints/epoch\=29/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg863_sauvola_rlsa_unet16_loss_no_weights_30epoch/2023-04-21/04-23-36/checkpoints/epoch\=28/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg863_sauvola_rlsa_unet16_loss_no_weights_30epoch/2023-04-21/04-28-03/checkpoints/epoch\=28/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg863_sauvola_rlsa_unet16_loss_no_weights_30epoch/2023-04-21/04-32-28/checkpoints/epoch\=29/backbone.pth"
)

for j in "${!weights_30[@]}"; do
  params="experiment=fine_tune_csg863_polygon_unet.yaml
          trainer.devices=${devices}
          mode=hip.yaml
          +model.backbone.path_to_weights=${weights_30[$j]}
          name=FT_sauvola_rlsa_csg863_polygon_unet16_loss_no_weights_30pt_100e
          logger.wandb.tags=[unet16,csg863,polygon,3-classes,fine-tune,100-epochs,no-weights,sauvola_rlsa,30-epoch-pt,with_header]
          logger.wandb.project=hip
          logger.wandb.group=fine-tune-sauvola_rlsa-3cl-polygon-csg863-30pt"
  python run.py ${params}
  #    echo ${params}
done

weights_40=("/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg863_sauvola_rlsa_unet16_loss_no_weights_40epoch/2023-04-21/04-36-55/checkpoints/epoch\=32/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg863_sauvola_rlsa_unet16_loss_no_weights_40epoch/2023-04-21/04-42-36/checkpoints/epoch\=38/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg863_sauvola_rlsa_unet16_loss_no_weights_40epoch/2023-04-21/04-48-18/checkpoints/epoch\=39/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg863_sauvola_rlsa_unet16_loss_no_weights_40epoch/2023-04-21/04-53-53/checkpoints/epoch\=38/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg863_sauvola_rlsa_unet16_loss_no_weights_40epoch/2023-04-21/04-59-33/checkpoints/epoch\=38/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg863_sauvola_rlsa_unet16_loss_no_weights_40epoch/2023-04-21/05-05-12/checkpoints/epoch\=38/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg863_sauvola_rlsa_unet16_loss_no_weights_40epoch/2023-04-21/05-10-54/checkpoints/epoch\=37/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg863_sauvola_rlsa_unet16_loss_no_weights_40epoch/2023-04-21/05-16-31/checkpoints/epoch\=38/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg863_sauvola_rlsa_unet16_loss_no_weights_40epoch/2023-04-21/05-22-08/checkpoints/epoch\=33/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg863_sauvola_rlsa_unet16_loss_no_weights_40epoch/2023-04-21/05-27-48/checkpoints/epoch\=32/backbone.pth"
)

for j in "${!weights_40[@]}"; do
  params="experiment=fine_tune_csg863_polygon_unet.yaml
          trainer.devices=${devices}
          mode=hip.yaml
          +model.backbone.path_to_weights=${weights_40[$j]}
          name=FT_sauvola_rlsa_csg863_polygon_unet16_loss_no_weights_40pt_100e
          logger.wandb.tags=[unet16,csg863,polygon,3-classes,fine-tune,100-epochs,no-weights,sauvola_rlsa,40-epoch-pt,with_header]
          logger.wandb.project=hip
          logger.wandb.group=fine-tune-sauvola_rlsa-3cl-polygon-csg863-40pt"
  python run.py ${params}
  #    echo ${params}
done

weights_50=("/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg863_sauvola_rlsa_unet16_loss_no_weights_50epoch/2023-04-21/05-33-29/checkpoints/epoch\=33/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg863_sauvola_rlsa_unet16_loss_no_weights_50epoch/2023-04-21/05-40-19/checkpoints/epoch\=37/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg863_sauvola_rlsa_unet16_loss_no_weights_50epoch/2023-04-21/05-47-09/checkpoints/epoch\=49/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg863_sauvola_rlsa_unet16_loss_no_weights_50epoch/2023-04-21/05-53-56/checkpoints/epoch\=38/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg863_sauvola_rlsa_unet16_loss_no_weights_50epoch/2023-04-21/06-00-46/checkpoints/epoch\=47/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg863_sauvola_rlsa_unet16_loss_no_weights_50epoch/2023-04-21/06-07-44/checkpoints/epoch\=47/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg863_sauvola_rlsa_unet16_loss_no_weights_50epoch/2023-04-21/06-14-36/checkpoints/epoch\=45/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg863_sauvola_rlsa_unet16_loss_no_weights_50epoch/2023-04-21/06-21-31/checkpoints/epoch\=45/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg863_sauvola_rlsa_unet16_loss_no_weights_50epoch/2023-04-21/06-28-25/checkpoints/epoch\=49/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg863_sauvola_rlsa_unet16_loss_no_weights_50epoch/2023-04-21/06-35-13/checkpoints/epoch\=48/backbone.pth"
)

for j in "${!weights_50[@]}"; do
  params="experiment=fine_tune_csg863_polygon_unet.yaml
          trainer.devices=${devices}
          mode=hip.yaml
          +model.backbone.path_to_weights=${weights_50[$j]}
          name=FT_sauvola_rlsa_csg863_polygon_unet16_loss_no_weights_50pt_100e
          logger.wandb.tags=[unet16,csg863,polygon,3-classes,fine-tune,100-epochs,no-weights,sauvola_rlsa,50-epoch-pt,with_header]
          logger.wandb.project=hip
          logger.wandb.group=fine-tune-sauvola_rlsa-3cl-polygon-csg863-50pt"
  python run.py ${params}
  #    echo ${params}
done
