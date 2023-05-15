#!/usr/bin/env bash

set -e

weights_10=("/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_new_rlsa_cb55_sauvola_unet_loss_no_weights_10epoch/2023-04-11/07-20-10/checkpoints/epoch\=6/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_new_rlsa_cb55_sauvola_unet_loss_no_weights_10epoch/2023-04-11/07-24-59/checkpoints/epoch\=8/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_new_rlsa_cb55_sauvola_unet_loss_no_weights_10epoch/2023-04-11/07-29-46/checkpoints/epoch\=9/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_new_rlsa_cb55_sauvola_unet_loss_no_weights_10epoch/2023-04-11/07-34-27/checkpoints/epoch\=6/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_new_rlsa_cb55_sauvola_unet_loss_no_weights_10epoch/2023-04-11/07-39-11/checkpoints/epoch\=7/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_new_rlsa_cb55_sauvola_unet_loss_no_weights_10epoch/2023-04-11/07-43-52/checkpoints/epoch\=7/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_new_rlsa_cb55_sauvola_unet_loss_no_weights_10epoch/2023-04-11/07-48-34/checkpoints/epoch\=6/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_new_rlsa_cb55_sauvola_unet_loss_no_weights_10epoch/2023-04-11/07-53-21/checkpoints/epoch\=5/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_new_rlsa_cb55_sauvola_unet_loss_no_weights_10epoch/2023-04-11/07-58-09/checkpoints/epoch\=9/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_new_rlsa_cb55_sauvola_unet_loss_no_weights_10epoch/2023-04-11/08-02-56/checkpoints/epoch\=5/backbone.pth"
)

devices="[0,1,2,3]"

for j in "${!weights_10[@]}"; do
  params="experiment=fine_tune_cb55_polygon_unet.yaml
          trainer.devices=${devices}
          mode=hip.yaml
          +model.backbone.path_to_weights=${weights_10[$j]}
          name=FT_sauvola_rlsa_new_3cl_cb55_polygon_unet_loss_no_weights_10pt_100e
          logger.wandb.project=hip
          logger.wandb.tags=[unet,cb55,polygon,3-classes,fine-tune,100-epochs,no-weights,3cl_new,rlsa_new_3cl,new_heuristic,10-epoch-pt]
          logger.wandb.group=fine-tune-sauvola-rlsa-new-3cl-polygon-cb55-10pt"
  python run.py ${params}
done

weights_20=("/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_new_rlsa_cb55_sauvola_unet_loss_no_weights_20epoch/2023-04-11/08-07-44/checkpoints/epoch\=12/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_new_rlsa_cb55_sauvola_unet_loss_no_weights_20epoch/2023-04-11/08-15-52/checkpoints/epoch\=9/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_new_rlsa_cb55_sauvola_unet_loss_no_weights_20epoch/2023-04-11/08-23-48/checkpoints/epoch\=16/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_new_rlsa_cb55_sauvola_unet_loss_no_weights_20epoch/2023-04-11/08-31-51/checkpoints/epoch\=8/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_new_rlsa_cb55_sauvola_unet_loss_no_weights_20epoch/2023-04-11/08-39-50/checkpoints/epoch\=13/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_new_rlsa_cb55_sauvola_unet_loss_no_weights_20epoch/2023-04-11/08-47-54/checkpoints/epoch\=13/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_new_rlsa_cb55_sauvola_unet_loss_no_weights_20epoch/2023-04-11/08-55-52/checkpoints/epoch\=16/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_new_rlsa_cb55_sauvola_unet_loss_no_weights_20epoch/2023-04-11/09-03-54/checkpoints/epoch\=14/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_new_rlsa_cb55_sauvola_unet_loss_no_weights_20epoch/2023-04-11/09-12-03/checkpoints/epoch\=14/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_new_rlsa_cb55_sauvola_unet_loss_no_weights_20epoch/2023-04-11/09-20-00/checkpoints/epoch\=19/backbone.pth"
)

for j in "${!weights_20[@]}"; do
  params="experiment=fine_tune_cb55_polygon_unet.yaml
          trainer.devices=${devices}
          mode=hip.yaml
          +model.backbone.path_to_weights=${weights_20[$j]}
          name=FT_sauvola_rlsa_new_3cl_cb55_polygon_unet_loss_no_weights_20pt_100e
          logger.wandb.project=hip
          logger.wandb.tags=[unet,cb55,polygon,3-classes,fine-tune,100-epochs,no-weights,3cl_new,rlsa_new_3cl,new_heuristic,20-epoch-pt]
          logger.wandb.group=fine-tune-sauvola-rlsa-new-3cl-polygon-cb55-20pt"
  python run.py ${params}
  #    echo ${params}
done

weights_30=("/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_new_rlsa_cb55_sauvola_unet_loss_no_weights_30epoch/2023-04-11/09-28-03/checkpoints/epoch\=14/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_new_rlsa_cb55_sauvola_unet_loss_no_weights_30epoch/2023-04-11/09-39-16/checkpoints/epoch\=18/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_new_rlsa_cb55_sauvola_unet_loss_no_weights_30epoch/2023-04-11/09-50-31/checkpoints/epoch\=19/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_new_rlsa_cb55_sauvola_unet_loss_no_weights_30epoch/2023-04-11/10-01-45/checkpoints/epoch\=28/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_new_rlsa_cb55_sauvola_unet_loss_no_weights_30epoch/2023-04-11/10-13-14/checkpoints/epoch\=23/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_new_rlsa_cb55_sauvola_unet_loss_no_weights_30epoch/2023-04-11/10-24-29/checkpoints/epoch\=19/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_new_rlsa_cb55_sauvola_unet_loss_no_weights_30epoch/2023-04-11/10-35-43/checkpoints/epoch\=15/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_new_rlsa_cb55_sauvola_unet_loss_no_weights_30epoch/2023-04-11/10-47-00/checkpoints/epoch\=21/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_new_rlsa_cb55_sauvola_unet_loss_no_weights_30epoch/2023-04-11/10-58-20/checkpoints/epoch\=25/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_new_rlsa_cb55_sauvola_unet_loss_no_weights_30epoch/2023-04-11/11-09-35/checkpoints/epoch\=28/backbone.pth"
)

for j in "${!weights_30[@]}"; do
  params="experiment=fine_tune_cb55_polygon_unet.yaml
          trainer.devices=${devices}
          mode=hip.yaml
          +model.backbone.path_to_weights=${weights_30[$j]}
          name=FT_sauvola_rlsa_new_3cl_cb55_polygon_unet_loss_no_weights_30pt_100e
          logger.wandb.project=hip
          logger.wandb.tags=[unet,cb55,polygon,3-classes,fine-tune,100-epochs,no-weights,3cl_new,rlsa_new_3cl,new_heuristic,30-epoch-pt]
          logger.wandb.group=fine-tune-sauvola-rlsa-new-3cl-polygon-cb55-30pt"
  python run.py ${params}
done

weights_40=("/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_new_rlsa_cb55_sauvola_unet_loss_no_weights_40epoch/2023-04-11/11-20-56/checkpoints/epoch\=18/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_new_rlsa_cb55_sauvola_unet_loss_no_weights_40epoch/2023-04-11/11-35-25/checkpoints/epoch\=35/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_new_rlsa_cb55_sauvola_unet_loss_no_weights_40epoch/2023-04-11/11-49-52/checkpoints/epoch\=18/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_new_rlsa_cb55_sauvola_unet_loss_no_weights_40epoch/2023-04-11/12-04-37/checkpoints/epoch\=28/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_new_rlsa_cb55_sauvola_unet_loss_no_weights_40epoch/2023-04-11/12-19-13/checkpoints/epoch\=14/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_new_rlsa_cb55_sauvola_unet_loss_no_weights_40epoch/2023-04-11/12-33-54/checkpoints/epoch\=14/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_new_rlsa_cb55_sauvola_unet_loss_no_weights_40epoch/2023-04-11/12-48-45/checkpoints/epoch\=26/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_new_rlsa_cb55_sauvola_unet_loss_no_weights_40epoch/2023-04-11/13-03-12/checkpoints/epoch\=23/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_new_rlsa_cb55_sauvola_unet_loss_no_weights_40epoch/2023-04-11/13-18-09/checkpoints/epoch\=32/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/3cl_new_rlsa_cb55_sauvola_unet_loss_no_weights_40epoch/2023-04-11/13-32-41/checkpoints/epoch\=23/backbone.pth"
)

for j in "${!weights_40[@]}"; do
  params="experiment=fine_tune_cb55_polygon_unet.yaml
          trainer.devices=${devices}
          mode=hip.yaml
          +model.backbone.path_to_weights=${weights_40[$j]}
          name=FT_sauvola_rlsa_new_3cl_cb55_polygon_unet_loss_no_weights_40pt_100e
          logger.wandb.project=hip
          logger.wandb.tags=[unet,cb55,polygon,3-classes,fine-tune,100-epochs,no-weights,3cl_new,rlsa_new_3cl,new_heuristic,40-epoch-pt]
          logger.wandb.group=fine-tune-sauvola-rlsa-new-3cl-polygon-cb55-40pt"
  python run.py ${params}
done

weights_50=("/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/3cl_new_rlsa_cb55_sauvola_unet_loss_no_weights_50epoch/2023-02-07/19-43-21/checkpoints/epoch\=21/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/3cl_new_rlsa_cb55_sauvola_unet_loss_no_weights_50epoch/2023-02-07/19-54-52/checkpoints/epoch\=34/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/3cl_new_rlsa_cb55_sauvola_unet_loss_no_weights_50epoch/2023-02-07/20-06-20/checkpoints/epoch\=20/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/3cl_new_rlsa_cb55_sauvola_unet_loss_no_weights_50epoch/2023-02-07/20-17-51/checkpoints/epoch\=14/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/3cl_new_rlsa_cb55_sauvola_unet_loss_no_weights_50epoch/2023-02-07/20-29-21/checkpoints/epoch\=17/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/3cl_new_rlsa_cb55_sauvola_unet_loss_no_weights_50epoch/2023-02-07/20-40-54/checkpoints/epoch\=44/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/3cl_new_rlsa_cb55_sauvola_unet_loss_no_weights_50epoch/2023-02-07/20-52-35/checkpoints/epoch\=26/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/3cl_new_rlsa_cb55_sauvola_unet_loss_no_weights_50epoch/2023-02-07/21-04-12/checkpoints/epoch\=15/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/3cl_new_rlsa_cb55_sauvola_unet_loss_no_weights_50epoch/2023-02-07/21-15-44/checkpoints/epoch\=27/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/3cl_new_rlsa_cb55_sauvola_unet_loss_no_weights_50epoch/2023-02-07/21-27-20/checkpoints/epoch\=46/backbone.pth"
)

for j in "${!weights_50[@]}"; do
  params="experiment=fine_tune_cb55_polygon_unet.yaml
          trainer.devices=${devices}
          mode=hip.yaml
          +model.backbone.path_to_weights=${weights_50[$j]}
          name=FT_sauvola_rlsa_new_3cl_cb55_polygon_unet_loss_no_weights_50pt_100e
          logger.wandb.project=hip
          logger.wandb.tags=[unet,cb55,polygon,3-classes,fine-tune,100-epochs,no-weights,3cl_new,rlsa_new_3cl,new_heuristic,50-epoch-pt]
          logger.wandb.group=fine-tune-sauvola-rlsa-new-3cl-polygon-cb55-50pt"
  python run.py ${params}

done
