#!/usr/bin/env bash

set -e

weights_10=("/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/rlsa_csg863_sauvola_unet_loss_no_weights_10epoch/2023-04-04/06-30-21/checkpoints/epoch\=7/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/rlsa_csg863_sauvola_unet_loss_no_weights_10epoch/2023-04-04/06-35-20/checkpoints/epoch\=9/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/rlsa_csg863_sauvola_unet_loss_no_weights_10epoch/2023-04-04/06-40-26/checkpoints/epoch\=8/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/rlsa_csg863_sauvola_unet_loss_no_weights_10epoch/2023-04-04/06-45-28/checkpoints/epoch\=9/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/rlsa_csg863_sauvola_unet_loss_no_weights_10epoch/2023-04-04/06-50-30/checkpoints/epoch\=8/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/rlsa_csg863_sauvola_unet_loss_no_weights_10epoch/2023-04-04/06-55-30/checkpoints/epoch\=9/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/rlsa_csg863_sauvola_unet_loss_no_weights_10epoch/2023-04-04/07-00-36/checkpoints/epoch\=9/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/rlsa_csg863_sauvola_unet_loss_no_weights_10epoch/2023-04-04/07-05-40/checkpoints/epoch\=7/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/rlsa_csg863_sauvola_unet_loss_no_weights_10epoch/2023-04-04/07-10-48/checkpoints/epoch\=9/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/rlsa_csg863_sauvola_unet_loss_no_weights_10epoch/2023-04-04/07-15-54/checkpoints/epoch\=6/backbone.pth"
)

devices="[0,1,2,3]"

for j in "${!weights_10[@]}"; do
  params="experiment=fine_tune_csg863_polygon_unet.yaml
          trainer.devices=${devices}
          mode=hip.yaml
          +model.backbone.path_to_weights=${weights_10[$j]}
          name=FT_sauvola_rlsa_csg863_polygon_unet_loss_no_weights_10pt_100e
          logger.wandb.tags=[unet,csg863,polygon,3-classes,fine-tune,100-epochs,no-weights,sauvola_rlsa,10-epoch-pt,with_header]
          logger.wandb.project=hip
          logger.wandb.group=fine-tune-sauvola_rlsa-3cl-polygon-csg863-10pt"
  python run.py ${params}
  #    echo ${params}
done

weights_20=("/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/rlsa_csg863_sauvola_unet_loss_no_weights_20epoch/2023-04-04/07-20-54/checkpoints/epoch\=17/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/rlsa_csg863_sauvola_unet_loss_no_weights_20epoch/2023-04-04/07-29-32/checkpoints/epoch\=19/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/rlsa_csg863_sauvola_unet_loss_no_weights_20epoch/2023-04-04/07-38-11/checkpoints/epoch\=17/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/rlsa_csg863_sauvola_unet_loss_no_weights_20epoch/2023-04-04/07-46-50/checkpoints/epoch\=16/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/rlsa_csg863_sauvola_unet_loss_no_weights_20epoch/2023-04-04/07-55-29/checkpoints/epoch\=19/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/rlsa_csg863_sauvola_unet_loss_no_weights_20epoch/2023-04-04/08-04-04/checkpoints/epoch\=14/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/rlsa_csg863_sauvola_unet_loss_no_weights_20epoch/2023-04-04/08-12-35/checkpoints/epoch\=17/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/rlsa_csg863_sauvola_unet_loss_no_weights_20epoch/2023-04-04/08-21-10/checkpoints/epoch\=19/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/rlsa_csg863_sauvola_unet_loss_no_weights_20epoch/2023-04-04/08-29-52/checkpoints/epoch\=14/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/rlsa_csg863_sauvola_unet_loss_no_weights_20epoch/2023-04-04/08-38-35/checkpoints/epoch\=12/backbone.pth"
)

for j in "${!weights_20[@]}"; do
  params="experiment=fine_tune_csg863_polygon_unet.yaml
          trainer.devices=${devices}
          mode=hip.yaml
          +model.backbone.path_to_weights=${weights_20[$j]}
          name=FT_sauvola_rlsa_csg863_polygon_unet_loss_no_weights_20pt_100e
          logger.wandb.tags=[unet,csg863,polygon,3-classes,fine-tune,100-epochs,no-weights,sauvola_rlsa,20-epoch-pt,with_header]
          logger.wandb.project=hip
          logger.wandb.group=fine-tune-sauvola_rlsa-3cl-polygon-csg863-20pt"
  python run.py ${params}
  #    echo ${params}
done

weights_30=("/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/rlsa_csg863_sauvola_unet_loss_no_weights_30epoch/2023-04-04/08-47-12/checkpoints/epoch\=29/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/rlsa_csg863_sauvola_unet_loss_no_weights_30epoch/2023-04-04/08-59-24/checkpoints/epoch\=19/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/rlsa_csg863_sauvola_unet_loss_no_weights_30epoch/2023-04-04/09-11-36/checkpoints/epoch\=28/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/rlsa_csg863_sauvola_unet_loss_no_weights_30epoch/2023-04-04/09-23-55/checkpoints/epoch\=29/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/rlsa_csg863_sauvola_unet_loss_no_weights_30epoch/2023-04-04/09-36-08/checkpoints/epoch\=27/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/rlsa_csg863_sauvola_unet_loss_no_weights_30epoch/2023-04-04/09-48-32/checkpoints/epoch\=24/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/rlsa_csg863_sauvola_unet_loss_no_weights_30epoch/2023-04-04/10-00-51/checkpoints/epoch\=26/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/rlsa_csg863_sauvola_unet_loss_no_weights_30epoch/2023-04-04/10-13-13/checkpoints/epoch\=18/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/rlsa_csg863_sauvola_unet_loss_no_weights_30epoch/2023-04-04/10-25-37/checkpoints/epoch\=20/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/rlsa_csg863_sauvola_unet_loss_no_weights_30epoch/2023-04-04/10-37-49/checkpoints/epoch\=19/backbone.pth"
)

for j in "${!weights_30[@]}"; do
  params="experiment=fine_tune_csg863_polygon_unet.yaml
          trainer.devices=${devices}
          mode=hip.yaml
          +model.backbone.path_to_weights=${weights_30[$j]}
          name=FT_sauvola_rlsa_csg863_polygon_unet_loss_no_weights_30pt_100e
          logger.wandb.tags=[unet,csg863,polygon,3-classes,fine-tune,100-epochs,no-weights,sauvola_rlsa,30-epoch-pt,with_header]
          logger.wandb.project=hip
          logger.wandb.group=fine-tune-sauvola_rlsa-3cl-polygon-csg863-30pt"
  python run.py ${params}
  #    echo ${params}
done

weights_40=("/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/rlsa_csg863_sauvola_unet_loss_no_weights_40epoch/2023-04-04/10-50-10/checkpoints/epoch\=32/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/rlsa_csg863_sauvola_unet_loss_no_weights_40epoch/2023-04-04/11-05-58/checkpoints/epoch\=34/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/rlsa_csg863_sauvola_unet_loss_no_weights_40epoch/2023-04-04/11-21-45/checkpoints/epoch\=31/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/rlsa_csg863_sauvola_unet_loss_no_weights_40epoch/2023-04-04/11-37-36/checkpoints/epoch\=32/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/rlsa_csg863_sauvola_unet_loss_no_weights_40epoch/2023-04-04/11-53-39/checkpoints/epoch\=26/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/rlsa_csg863_sauvola_unet_loss_no_weights_40epoch/2023-04-04/12-09-21/checkpoints/epoch\=36/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/rlsa_csg863_sauvola_unet_loss_no_weights_40epoch/2023-04-04/12-25-31/checkpoints/epoch\=37/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/rlsa_csg863_sauvola_unet_loss_no_weights_40epoch/2023-04-04/12-41-06/checkpoints/epoch\=34/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/rlsa_csg863_sauvola_unet_loss_no_weights_40epoch/2023-04-04/12-56-57/checkpoints/epoch\=35/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/rlsa_csg863_sauvola_unet_loss_no_weights_40epoch/2023-04-04/13-12-51/checkpoints/epoch\=29/backbone.pth"
)

for j in "${!weights_40[@]}"; do
  params="experiment=fine_tune_csg863_polygon_unet.yaml
          trainer.devices=${devices}
          mode=hip.yaml
          +model.backbone.path_to_weights=${weights_40[$j]}
          name=FT_sauvola_rlsa_csg863_polygon_unet_loss_no_weights_40pt_100e
          logger.wandb.tags=[unet,csg863,polygon,3-classes,fine-tune,100-epochs,no-weights,sauvola_rlsa,40-epoch-pt,with_header]
          logger.wandb.project=hip
          logger.wandb.group=fine-tune-sauvola_rlsa-3cl-polygon-csg863-40pt"
  python run.py ${params}
  #    echo ${params}
done

weights_50=("/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/rlsa_csg863_sauvola_unet_loss_no_weights_50epoch/2023-04-04/13-28-32/checkpoints/epoch\=46/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/rlsa_csg863_sauvola_unet_loss_no_weights_50epoch/2023-04-04/13-47-51/checkpoints/epoch\=46/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/rlsa_csg863_sauvola_unet_loss_no_weights_50epoch/2023-04-04/14-06-59/checkpoints/epoch\=48/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/rlsa_csg863_sauvola_unet_loss_no_weights_50epoch/2023-04-04/14-26-33/checkpoints/epoch\=36/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/rlsa_csg863_sauvola_unet_loss_no_weights_50epoch/2023-04-04/14-46-08/checkpoints/epoch\=38/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/rlsa_csg863_sauvola_unet_loss_no_weights_50epoch/2023-04-04/15-05-21/checkpoints/epoch\=34/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/rlsa_csg863_sauvola_unet_loss_no_weights_50epoch/2023-04-04/15-24-34/checkpoints/epoch\=31/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/rlsa_csg863_sauvola_unet_loss_no_weights_50epoch/2023-04-04/15-44-02/checkpoints/epoch\=31/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/rlsa_csg863_sauvola_unet_loss_no_weights_50epoch/2023-04-04/16-03-41/checkpoints/epoch\=29/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/rlsa_csg863_sauvola_unet_loss_no_weights_50epoch/2023-04-04/16-23-30/checkpoints/epoch\=34/backbone.pth"
)

for j in "${!weights_50[@]}"; do
  params="experiment=fine_tune_csg863_polygon_unet.yaml
          trainer.devices=${devices}
          mode=hip.yaml
          +model.backbone.path_to_weights=${weights_50[$j]}
          name=FT_sauvola_rlsa_csg863_polygon_unet_loss_no_weights_50pt_100e
          logger.wandb.tags=[unet,csg863,polygon,3-classes,fine-tune,100-epochs,no-weights,sauvola_rlsa,50-epoch-pt,with_header]
          logger.wandb.project=hip
          logger.wandb.group=fine-tune-sauvola_rlsa-3cl-polygon-csg863-50pt"
  python run.py ${params}
  #    echo ${params}
done
