#!/usr/bin/env bash

set -e

weights_10=("/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg863_1152_1728_sauvola_unet_loss_no_weights_10epoch/2023-03-28/12-45-23/checkpoints/epoch\=9/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg863_1152_1728_sauvola_unet_loss_no_weights_10epoch/2023-03-28/12-51-52/checkpoints/epoch\=9/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg863_1152_1728_sauvola_unet_loss_no_weights_10epoch/2023-03-28/12-58-20/checkpoints/epoch\=9/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg863_1152_1728_sauvola_unet_loss_no_weights_10epoch/2023-03-28/13-04-55/checkpoints/epoch\=9/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg863_1152_1728_sauvola_unet_loss_no_weights_10epoch/2023-03-28/13-11-20/checkpoints/epoch\=8/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg863_1152_1728_sauvola_unet_loss_no_weights_10epoch/2023-03-28/13-17-55/checkpoints/epoch\=8/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg863_1152_1728_sauvola_unet_loss_no_weights_10epoch/2023-03-28/13-24-31/checkpoints/epoch\=8/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg863_1152_1728_sauvola_unet_loss_no_weights_10epoch/2023-03-28/13-30-56/checkpoints/epoch\=9/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg863_1152_1728_sauvola_unet_loss_no_weights_10epoch/2023-03-28/13-37-22/checkpoints/epoch\=9/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg863_1152_1728_sauvola_unet_loss_no_weights_10epoch/2023-03-28/13-43-52/checkpoints/epoch\=9/backbone.pth"
)

devices="[0,1,2,3]"

for j in "${!weights_10[@]}"; do
  params="experiment=fine_tune_csg863_polygon_unet.yaml
          trainer.devices=${devices}
          mode=hip.yaml
          +model.backbone.path_to_weights=${weights_10[$j]}
          name=FT_sauvola_csg863_polygon_unet_loss_no_weights_10pt_100e
          logger.wandb.tags=[unet,csg863,polygon,3-classes,fine-tune,100-epochs,no-weights,sauvola,10-epoch-pt,with_header]
          logger.wandb.project=hip
          logger.wandb.group=fine-tune-sauvola-3cl-polygon-csg863-10pt"
  python run.py ${params}
  #    echo ${params}
done

weights_20=("/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg863_1152_1728_sauvola_unet_loss_no_weights_20epoch/2023-03-28/13-50-22/checkpoints/epoch\=18/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg863_1152_1728_sauvola_unet_loss_no_weights_20epoch/2023-03-28/14-01-40/checkpoints/epoch\=17/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg863_1152_1728_sauvola_unet_loss_no_weights_20epoch/2023-03-28/14-12-56/checkpoints/epoch\=16/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg863_1152_1728_sauvola_unet_loss_no_weights_20epoch/2023-03-28/14-24-09/checkpoints/epoch\=19/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg863_1152_1728_sauvola_unet_loss_no_weights_20epoch/2023-03-28/14-35-25/checkpoints/epoch\=19/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg863_1152_1728_sauvola_unet_loss_no_weights_20epoch/2023-03-28/14-46-38/checkpoints/epoch\=19/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg863_1152_1728_sauvola_unet_loss_no_weights_20epoch/2023-03-28/14-57-56/checkpoints/epoch\=13/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg863_1152_1728_sauvola_unet_loss_no_weights_20epoch/2023-03-28/15-09-24/checkpoints/epoch\=14/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg863_1152_1728_sauvola_unet_loss_no_weights_20epoch/2023-03-28/15-20-39/checkpoints/epoch\=19/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg863_1152_1728_sauvola_unet_loss_no_weights_20epoch/2023-03-28/15-32-03/checkpoints/epoch\=15/backbone.pth"
)

for j in "${!weights_20[@]}"; do
  params="experiment=fine_tune_csg863_polygon_unet.yaml
          trainer.devices=${devices}
          mode=hip.yaml
          +model.backbone.path_to_weights=${weights_20[$j]}
          name=FT_sauvola_csg863_polygon_unet_loss_no_weights_20pt_100e
          logger.wandb.tags=[unet,csg863,polygon,3-classes,fine-tune,100-epochs,no-weights,sauvola,20-epoch-pt,with_header]
          logger.wandb.project=hip
          logger.wandb.group=fine-tune-sauvola-3cl-polygon-csg863-20pt"
  python run.py ${params}
  #    echo ${params}
done

weights_30=("/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg863_1152_1728_sauvola_unet_loss_no_weights_30epoch/2023-03-28/15-43-24/checkpoints/epoch\=29/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg863_1152_1728_sauvola_unet_loss_no_weights_30epoch/2023-03-28/15-59-27/checkpoints/epoch\=22/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg863_1152_1728_sauvola_unet_loss_no_weights_30epoch/2023-03-28/16-15-26/checkpoints/epoch\=25/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg863_1152_1728_sauvola_unet_loss_no_weights_30epoch/2023-03-28/16-31-28/checkpoints/epoch\=26/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg863_1152_1728_sauvola_unet_loss_no_weights_30epoch/2023-03-28/16-47-36/checkpoints/epoch\=20/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg863_1152_1728_sauvola_unet_loss_no_weights_30epoch/2023-03-28/17-03-36/checkpoints/epoch\=27/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg863_1152_1728_sauvola_unet_loss_no_weights_30epoch/2023-03-28/17-19-59/checkpoints/epoch\=23/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg863_1152_1728_sauvola_unet_loss_no_weights_30epoch/2023-03-28/17-36-18/checkpoints/epoch\=27/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg863_1152_1728_sauvola_unet_loss_no_weights_30epoch/2023-03-28/17-52-15/checkpoints/epoch\=26/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg863_1152_1728_sauvola_unet_loss_no_weights_30epoch/2023-03-28/18-08-12/checkpoints/epoch\=28/backbone.pth"
)

for j in "${!weights_30[@]}"; do
  params="experiment=fine_tune_csg863_polygon_unet.yaml
          trainer.devices=${devices}
          mode=hip.yaml
          +model.backbone.path_to_weights=${weights_30[$j]}
          name=FT_sauvola_csg863_polygon_unet_loss_no_weights_30pt_100e
          logger.wandb.tags=[unet,csg863,polygon,3-classes,fine-tune,100-epochs,no-weights,sauvola,30-epoch-pt,with_header]
          logger.wandb.project=hip
          logger.wandb.group=fine-tune-sauvola-3cl-polygon-csg863-30pt"
  python run.py ${params}
  #    echo ${params}
done

weights_40=("/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg863_1152_1728_sauvola_unet_loss_no_weights_40epoch/2023-03-28/18-24-10/checkpoints/epoch\=39/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg863_1152_1728_sauvola_unet_loss_no_weights_40epoch/2023-03-28/18-44-54/checkpoints/epoch\=39/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg863_1152_1728_sauvola_unet_loss_no_weights_40epoch/2023-03-28/19-05-38/checkpoints/epoch\=33/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg863_1152_1728_sauvola_unet_loss_no_weights_40epoch/2023-03-28/19-26-27/checkpoints/epoch\=27/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg863_1152_1728_sauvola_unet_loss_no_weights_40epoch/2023-03-28/19-47-02/checkpoints/epoch\=39/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg863_1152_1728_sauvola_unet_loss_no_weights_40epoch/2023-03-28/20-07-58/checkpoints/epoch\=37/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg863_1152_1728_sauvola_unet_loss_no_weights_40epoch/2023-03-28/20-28-49/checkpoints/epoch\=33/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg863_1152_1728_sauvola_unet_loss_no_weights_40epoch/2023-03-28/20-49-28/checkpoints/epoch\=36/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg863_1152_1728_sauvola_unet_loss_no_weights_40epoch/2023-03-28/21-10-06/checkpoints/epoch\=36/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg863_1152_1728_sauvola_unet_loss_no_weights_40epoch/2023-03-28/21-30-55/checkpoints/epoch\=34/backbone.pth"
)

for j in "${!weights_40[@]}"; do
  params="experiment=fine_tune_csg863_polygon_unet.yaml
          trainer.devices=${devices}
          mode=hip.yaml
          +model.backbone.path_to_weights=${weights_40[$j]}
          name=FT_sauvola_csg863_polygon_unet_loss_no_weights_40pt_100e
          logger.wandb.tags=[unet,csg863,polygon,3-classes,fine-tune,100-epochs,no-weights,sauvola,40-epoch-pt,with_header]
          logger.wandb.project=hip
          logger.wandb.group=fine-tune-sauvola-3cl-polygon-csg863-40pt"
  python run.py ${params}
  #    echo ${params}
done

weights_50=("/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg863_1152_1728_sauvola_unet_loss_no_weights_50epoch/2023-03-28/21-52-04/checkpoints/epoch\=40/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg863_1152_1728_sauvola_unet_loss_no_weights_50epoch/2023-03-28/22-17-51/checkpoints/epoch\=48/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg863_1152_1728_sauvola_unet_loss_no_weights_50epoch/2023-03-28/22-43-42/checkpoints/epoch\=45/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg863_1152_1728_sauvola_unet_loss_no_weights_50epoch/2023-03-28/23-09-29/checkpoints/epoch\=47/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg863_1152_1728_sauvola_unet_loss_no_weights_50epoch/2023-03-28/23-35-33/checkpoints/epoch\=49/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg863_1152_1728_sauvola_unet_loss_no_weights_50epoch/2023-03-29/00-01-26/checkpoints/epoch\=48/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg863_1152_1728_sauvola_unet_loss_no_weights_50epoch/2023-03-29/00-26-52/checkpoints/epoch\=34/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg863_1152_1728_sauvola_unet_loss_no_weights_50epoch/2023-03-29/00-52-20/checkpoints/epoch\=41/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg863_1152_1728_sauvola_unet_loss_no_weights_50epoch/2023-03-29/01-17-46/checkpoints/epoch\=45/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_rlsa_csg863_1152_1728_sauvola_unet_loss_no_weights_50epoch/2023-03-29/01-43-12/checkpoints/epoch\=40/backbone.pth"
)

for j in "${!weights_50[@]}"; do
  params="experiment=fine_tune_csg863_polygon_unet.yaml
          trainer.devices=${devices}
          mode=hip.yaml
          +model.backbone.path_to_weights=${weights_50[$j]}
          name=FT_sauvola_csg863_polygon_unet_loss_no_weights_50pt_100e
          logger.wandb.tags=[unet,csg863,polygon,3-classes,fine-tune,100-epochs,no-weights,sauvola,50-epoch-pt,with_header]
          logger.wandb.project=hip
          logger.wandb.group=fine-tune-sauvola-3cl-polygon-csg863-50pt"
  python run.py ${params}
  #    echo ${params}
done
