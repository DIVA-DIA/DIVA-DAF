#!/usr/bin/env bash

set -e

weights_10=("/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg863_sauvola_unet16_loss_no_weights_10epoch/2023-04-20/22-07-45/checkpoints/epoch\=9/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg863_sauvola_unet16_loss_no_weights_10epoch/2023-04-20/22-10-20/checkpoints/epoch\=9/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg863_sauvola_unet16_loss_no_weights_10epoch/2023-04-20/22-12-56/checkpoints/epoch\=9/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg863_sauvola_unet16_loss_no_weights_10epoch/2023-04-20/22-15-33/checkpoints/epoch\=9/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg863_sauvola_unet16_loss_no_weights_10epoch/2023-04-20/22-18-12/checkpoints/epoch\=9/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg863_sauvola_unet16_loss_no_weights_10epoch/2023-04-20/22-20-52/checkpoints/epoch\=9/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg863_sauvola_unet16_loss_no_weights_10epoch/2023-04-20/22-23-28/checkpoints/epoch\=9/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg863_sauvola_unet16_loss_no_weights_10epoch/2023-04-20/22-26-07/checkpoints/epoch\=9/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg863_sauvola_unet16_loss_no_weights_10epoch/2023-04-20/22-28-45/checkpoints/epoch\=9/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg863_sauvola_unet16_loss_no_weights_10epoch/2023-04-20/22-31-21/checkpoints/epoch\=9/backbone.pth"
)

devices="[0,1,2,3]"

for j in "${!weights_10[@]}"; do
  params="experiment=fine_tune_csg863_polygon_unet.yaml
          trainer.devices=${devices}
          mode=hip.yaml
          +model.backbone.path_to_weights=${weights_10[$j]}
          name=FT_sauvola_csg863_polygon_unet16_loss_no_weights_10pt_100e
          logger.wandb.tags=[unet16,csg863,polygon,3-classes,fine-tune,100-epochs,no-weights,sauvola,10-epoch-pt,with_header]
          logger.wandb.project=hip
          logger.wandb.group=fine-tune-sauvola-3cl-polygon-csg863-10pt"
  python run.py ${params}
  #    echo ${params}
done

weights_20=("/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg863_sauvola_unet16_loss_no_weights_20epoch/2023-04-20/22-33-57/checkpoints/epoch\=18/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg863_sauvola_unet16_loss_no_weights_20epoch/2023-04-20/22-38-08/checkpoints/epoch\=18/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg863_sauvola_unet16_loss_no_weights_20epoch/2023-04-20/22-42-23/checkpoints/epoch\=18/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg863_sauvola_unet16_loss_no_weights_20epoch/2023-04-20/22-46-36/checkpoints/epoch\=19/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg863_sauvola_unet16_loss_no_weights_20epoch/2023-04-20/22-50-47/checkpoints/epoch\=19/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg863_sauvola_unet16_loss_no_weights_20epoch/2023-04-20/22-55-02/checkpoints/epoch\=18/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg863_sauvola_unet16_loss_no_weights_20epoch/2023-04-20/22-59-12/checkpoints/epoch\=17/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg863_sauvola_unet16_loss_no_weights_20epoch/2023-04-20/23-03-25/checkpoints/epoch\=18/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg863_sauvola_unet16_loss_no_weights_20epoch/2023-04-20/23-07-40/checkpoints/epoch\=19/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg863_sauvola_unet16_loss_no_weights_20epoch/2023-04-20/23-11-47/checkpoints/epoch\=19/backbone.pth"
)

for j in "${!weights_20[@]}"; do
  params="experiment=fine_tune_csg863_polygon_unet.yaml
          trainer.devices=${devices}
          mode=hip.yaml
          +model.backbone.path_to_weights=${weights_20[$j]}
          name=FT_sauvola_csg863_polygon_unet16_loss_no_weights_20pt_100e
          logger.wandb.tags=[unet16,csg863,polygon,3-classes,fine-tune,100-epochs,no-weights,sauvola,20-epoch-pt,with_header]
          logger.wandb.project=hip
          logger.wandb.group=fine-tune-sauvola-3cl-polygon-csg863-20pt"
  python run.py ${params}
  #    echo ${params}
done

weights_30=("/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg863_sauvola_unet16_loss_no_weights_30epoch/2023-04-20/23-15-58/checkpoints/epoch\=29/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg863_sauvola_unet16_loss_no_weights_30epoch/2023-04-20/23-21-44/checkpoints/epoch\=24/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg863_sauvola_unet16_loss_no_weights_30epoch/2023-04-20/23-27-29/checkpoints/epoch\=27/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg863_sauvola_unet16_loss_no_weights_30epoch/2023-04-20/23-33-20/checkpoints/epoch\=27/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg863_sauvola_unet16_loss_no_weights_30epoch/2023-04-20/23-39-09/checkpoints/epoch\=26/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg863_sauvola_unet16_loss_no_weights_30epoch/2023-04-20/23-44-56/checkpoints/epoch\=29/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg863_sauvola_unet16_loss_no_weights_30epoch/2023-04-20/23-50-44/checkpoints/epoch\=27/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg863_sauvola_unet16_loss_no_weights_30epoch/2023-04-20/23-56-33/checkpoints/epoch\=28/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg863_sauvola_unet16_loss_no_weights_30epoch/2023-04-21/00-02-21/checkpoints/epoch\=28/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg863_sauvola_unet16_loss_no_weights_30epoch/2023-04-21/00-08-06/checkpoints/epoch\=29/backbone.pth"
)

for j in "${!weights_30[@]}"; do
  params="experiment=fine_tune_csg863_polygon_unet.yaml
          trainer.devices=${devices}
          mode=hip.yaml
          +model.backbone.path_to_weights=${weights_30[$j]}
          name=FT_sauvola_csg863_polygon_unet16_loss_no_weights_30pt_100e
          logger.wandb.tags=[unet16,csg863,polygon,3-classes,fine-tune,100-epochs,no-weights,sauvola,30-epoch-pt,with_header]
          logger.wandb.project=hip
          logger.wandb.group=fine-tune-sauvola-3cl-polygon-csg863-30pt"
  python run.py ${params}
  #    echo ${params}
done

weights_40=("/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg863_sauvola_unet16_loss_no_weights_40epoch/2023-04-21/00-13-54/checkpoints/epoch\=35/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg863_sauvola_unet16_loss_no_weights_40epoch/2023-04-21/00-21-20/checkpoints/epoch\=39/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg863_sauvola_unet16_loss_no_weights_40epoch/2023-04-21/00-28-45/checkpoints/epoch\=36/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg863_sauvola_unet16_loss_no_weights_40epoch/2023-04-21/00-36-12/checkpoints/epoch\=36/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg863_sauvola_unet16_loss_no_weights_40epoch/2023-04-21/00-43-40/checkpoints/epoch\=36/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg863_sauvola_unet16_loss_no_weights_40epoch/2023-04-21/00-51-10/checkpoints/epoch\=38/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg863_sauvola_unet16_loss_no_weights_40epoch/2023-04-21/00-58-36/checkpoints/epoch\=37/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg863_sauvola_unet16_loss_no_weights_40epoch/2023-04-21/01-06-01/checkpoints/epoch\=32/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg863_sauvola_unet16_loss_no_weights_40epoch/2023-04-21/01-13-27/checkpoints/epoch\=39/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg863_sauvola_unet16_loss_no_weights_40epoch/2023-04-21/01-20-51/checkpoints/epoch\=36/backbone.pth"
)

for j in "${!weights_40[@]}"; do
  params="experiment=fine_tune_csg863_polygon_unet.yaml
          trainer.devices=${devices}
          mode=hip.yaml
          +model.backbone.path_to_weights=${weights_40[$j]}
          name=FT_sauvola_csg863_polygon_unet16_loss_no_weights_40pt_100e
          logger.wandb.tags=[unet16,csg863,polygon,3-classes,fine-tune,100-epochs,no-weights,sauvola,40-epoch-pt,with_header]
          logger.wandb.project=hip
          logger.wandb.group=fine-tune-sauvola-3cl-polygon-csg863-40pt"
  python run.py ${params}
  #    echo ${params}
done

weights_50=("/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg863_sauvola_unet16_loss_no_weights_50epoch/2023-04-21/01-28-16/checkpoints/epoch\=48/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg863_sauvola_unet16_loss_no_weights_50epoch/2023-04-21/01-37-15/checkpoints/epoch\=46/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg863_sauvola_unet16_loss_no_weights_50epoch/2023-04-21/01-46-19/checkpoints/epoch\=43/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg863_sauvola_unet16_loss_no_weights_50epoch/2023-04-21/01-55-23/checkpoints/epoch\=43/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg863_sauvola_unet16_loss_no_weights_50epoch/2023-04-21/02-04-29/checkpoints/epoch\=49/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg863_sauvola_unet16_loss_no_weights_50epoch/2023-04-21/02-13-26/checkpoints/epoch\=47/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg863_sauvola_unet16_loss_no_weights_50epoch/2023-04-21/02-22-28/checkpoints/epoch\=48/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg863_sauvola_unet16_loss_no_weights_50epoch/2023-04-21/02-31-40/checkpoints/epoch\=42/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg863_sauvola_unet16_loss_no_weights_50epoch/2023-04-21/02-40-42/checkpoints/epoch\=47/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_csg863_sauvola_unet16_loss_no_weights_50epoch/2023-04-21/02-49-44/checkpoints/epoch\=45/backbone.pth"
)

for j in "${!weights_50[@]}"; do
  params="experiment=fine_tune_csg863_polygon_unet.yaml
          trainer.devices=${devices}
          mode=hip.yaml
          +model.backbone.path_to_weights=${weights_50[$j]}
          name=FT_sauvola_csg863_polygon_unet16_loss_no_weights_50pt_100e
          logger.wandb.tags=[unet16,csg863,polygon,3-classes,fine-tune,100-epochs,no-weights,sauvola,50-epoch-pt,with_header]
          logger.wandb.project=hip
          logger.wandb.group=fine-tune-sauvola-3cl-polygon-csg863-50pt"
  python run.py ${params}
  #    echo ${params}
done
