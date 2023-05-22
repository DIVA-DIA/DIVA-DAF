#!/usr/bin/env bash

set -e

weights_10=("/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_heuristic_unet16_loss_no_weights_10epoch/2023-04-21/06-42-09/checkpoints/epoch\=9/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_heuristic_unet16_loss_no_weights_10epoch/2023-04-21/06-44-09/checkpoints/epoch\=9/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_heuristic_unet16_loss_no_weights_10epoch/2023-04-21/06-46-15/checkpoints/epoch\=9/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_heuristic_unet16_loss_no_weights_10epoch/2023-04-21/06-48-19/checkpoints/epoch\=7/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_heuristic_unet16_loss_no_weights_10epoch/2023-04-21/06-50-16/checkpoints/epoch\=8/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_heuristic_unet16_loss_no_weights_10epoch/2023-04-21/06-52-21/checkpoints/epoch\=8/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_heuristic_unet16_loss_no_weights_10epoch/2023-04-21/06-54-20/checkpoints/epoch\=9/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_heuristic_unet16_loss_no_weights_10epoch/2023-04-21/06-56-20/checkpoints/epoch\=9/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_heuristic_unet16_loss_no_weights_10epoch/2023-04-21/06-58-22/checkpoints/epoch\=6/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_heuristic_unet16_loss_no_weights_10epoch/2023-04-21/07-00-26/checkpoints/epoch\=8/backbone.pth"
)

devices="[0,1,2,3]"
#
#for j in "${!weights_10[@]}"; do
#  params_18="experiment=fine_tune_csg18_polygon_unet.yaml
#          trainer.devices=${devices}
#          mode=hip.yaml
#          +model.backbone.path_to_weights=${weights_10[$j]}
#          name=FT_sauvola_rlsa_new_3cl_csg18_polygon_unet16_loss_no_weights_10pt_100e
#          logger.wandb.project=hip
#          logger.wandb.tags=[unet16,csg18,polygon,3-classes,fine-tune,100-epochs,no-weights,3cl_new,rlsa_new_3cl,new_heuristic,10-epoch-pt,with_header]
#          logger.wandb.group=fine-tune-sauvola-rlsa-new-3cl-polygon-csg18-10pt"
#  python run.py ${params_18}
#
#  params_863="experiment=fine_tune_csg863_polygon_unet.yaml
#          trainer.devices=${devices}
#          mode=hip.yaml
#          +model.backbone.path_to_weights=${weights_10[$j]}
#          name=FT_sauvola_rlsa_new_3cl_csg863_polygon_unet16_loss_no_weights_10pt_100e
#          logger.wandb.project=hip
#          logger.wandb.tags=[unet16,csg863,polygon,3-classes,fine-tune,100-epochs,no-weights,3cl_new,rlsa_new_3cl,new_heuristic,10-epoch-pt,with_header]
#          logger.wandb.group=fine-tune-sauvola-rlsa-new-3cl-csg863-polygon-10pt"
#  python run.py ${params_863}
#  #    echo ${params}
#done

weights_20=("/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_heuristic_unet16_loss_no_weights_20epoch/2023-04-21/07-02-26/checkpoints/epoch\=10/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_heuristic_unet16_loss_no_weights_20epoch/2023-04-21/07-05-31/checkpoints/epoch\=17/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_heuristic_unet16_loss_no_weights_20epoch/2023-04-21/07-08-35/checkpoints/epoch\=14/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_heuristic_unet16_loss_no_weights_20epoch/2023-04-21/07-11-43/checkpoints/epoch\=19/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_heuristic_unet16_loss_no_weights_20epoch/2023-04-21/07-14-53/checkpoints/epoch\=17/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_heuristic_unet16_loss_no_weights_20epoch/2023-04-21/07-17-59/checkpoints/epoch\=16/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_heuristic_unet16_loss_no_weights_20epoch/2023-04-21/07-21-04/checkpoints/epoch\=19/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_heuristic_unet16_loss_no_weights_20epoch/2023-04-21/07-24-08/checkpoints/epoch\=14/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_heuristic_unet16_loss_no_weights_20epoch/2023-04-21/07-27-12/checkpoints/epoch\=17/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_heuristic_unet16_loss_no_weights_20epoch/2023-04-21/07-30-16/checkpoints/epoch\=14/backbone.pth"
)
#
#for j in "${!weights_20[@]}"; do
#  params_18="experiment=fine_tune_csg18_polygon_unet.yaml
#          trainer.devices=${devices}
#          mode=hip.yaml
#          +model.backbone.path_to_weights=${weights_20[$j]}
#          name=FT_sauvola_rlsa_new_3cl_csg18_polygon_unet16_loss_no_weights_20pt_100e
#          logger.wandb.project=hip
#          logger.wandb.tags=[unet16,csg18,polygon,3-classes,fine-tune,100-epochs,no-weights,3cl_new,rlsa_new_3cl,new_heuristic,20-epoch-pt,with_header]
#          logger.wandb.group=fine-tune-sauvola-rlsa-new-3cl-polygon-csg18-20pt"
#  python run.py ${params_18}
#
#  params_863="experiment=fine_tune_csg863_polygon_unet.yaml
#          trainer.devices=${devices}
#          mode=hip.yaml
#          +model.backbone.path_to_weights=${weights_20[$j]}
#          name=FT_sauvola_rlsa_new_3cl_csg863_polygon_unet16_loss_no_weights_20pt_100e
#          logger.wandb.project=hip
#          logger.wandb.tags=[unet16,csg863,polygon,3-classes,fine-tune,100-epochs,no-weights,3cl_new,rlsa_new_3cl,new_heuristic,20-epoch-pt,with_header]
#          logger.wandb.group=fine-tune-sauvola-rlsa-new-3cl-csg863-polygon-20pt"
#  python run.py ${params_863}
#  #    echo ${params}
#done

weights_30=("/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_heuristic_unet16_loss_no_weights_30epoch/2023-04-21/07-33-21/checkpoints/epoch\=27/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_heuristic_unet16_loss_no_weights_30epoch/2023-04-21/07-37-27/checkpoints/epoch\=22/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_heuristic_unet16_loss_no_weights_30epoch/2023-04-21/07-41-35/checkpoints/epoch\=17/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_heuristic_unet16_loss_no_weights_30epoch/2023-04-21/07-45-39/checkpoints/epoch\=20/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_heuristic_unet16_loss_no_weights_30epoch/2023-04-21/07-49-46/checkpoints/epoch\=28/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_heuristic_unet16_loss_no_weights_30epoch/2023-04-21/07-53-53/checkpoints/epoch\=18/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_heuristic_unet16_loss_no_weights_30epoch/2023-04-21/07-58-03/checkpoints/epoch\=24/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_heuristic_unet16_loss_no_weights_30epoch/2023-04-21/08-02-14/checkpoints/epoch\=15/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_heuristic_unet16_loss_no_weights_30epoch/2023-04-21/08-06-20/checkpoints/epoch\=26/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_heuristic_unet16_loss_no_weights_30epoch/2023-04-21/08-10-28/checkpoints/epoch\=19/backbone.pth"
)
#
#for j in "${!weights_30[@]}"; do
#  params_18="experiment=fine_tune_csg18_polygon_unet.yaml
#          trainer.devices=${devices}
#          mode=hip.yaml
#          +model.backbone.path_to_weights=${weights_30[$j]}
#          name=FT_sauvola_rlsa_new_3cl_csg18_polygon_unet16_loss_no_weights_30pt_100e
#          logger.wandb.project=hip
#          logger.wandb.tags=[unet16,csg18,polygon,3-classes,fine-tune,100-epochs,no-weights,3cl_new,rlsa_new_3cl,new_heuristic,30-epoch-pt,with_header]
#          logger.wandb.group=fine-tune-sauvola-rlsa-new-3cl-polygon-csg18-30pt"
#  python run.py ${params_18}
#
#  params_863="experiment=fine_tune_csg863_polygon_unet.yaml
#          trainer.devices=${devices}
#          mode=hip.yaml
#          +model.backbone.path_to_weights=${weights_30[$j]}
#          name=FT_sauvola_rlsa_new_3cl_csg863_polygon_unet16_loss_no_weights_30pt_100e
#          logger.wandb.project=hip
#          logger.wandb.tags=[unet16,csg863,polygon,3-classes,fine-tune,100-epochs,no-weights,3cl_new,rlsa_new_3cl,new_heuristic,30-epoch-pt,with_header]
#          logger.wandb.group=fine-tune-sauvola-rlsa-new-3cl-csg863-polygon-30pt"
#  python run.py ${params_863}
#  #    echo ${params}
#done

weights_40=("/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_heuristic_unet16_loss_no_weights_40epoch/2023-04-21/08-14-36/checkpoints/epoch\=26/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_heuristic_unet16_loss_no_weights_40epoch/2023-04-21/08-19-49/checkpoints/epoch\=38/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_heuristic_unet16_loss_no_weights_40epoch/2023-04-21/08-24-57/checkpoints/epoch\=28/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_heuristic_unet16_loss_no_weights_40epoch/2023-04-21/08-30-05/checkpoints/epoch\=27/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_heuristic_unet16_loss_no_weights_40epoch/2023-04-21/08-35-20/checkpoints/epoch\=23/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_heuristic_unet16_loss_no_weights_40epoch/2023-04-21/08-40-36/checkpoints/epoch\=26/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_heuristic_unet16_loss_no_weights_40epoch/2023-04-21/08-45-51/checkpoints/epoch\=32/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_heuristic_unet16_loss_no_weights_40epoch/2023-04-21/08-50-59/checkpoints/epoch\=24/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_heuristic_unet16_loss_no_weights_40epoch/2023-04-21/08-56-17/checkpoints/epoch\=26/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_heuristic_unet16_loss_no_weights_40epoch/2023-04-21/09-01-28/checkpoints/epoch\=27/backbone.pth"
)
#
#for j in "${!weights_40[@]}"; do
#  params_18="experiment=fine_tune_csg18_polygon_unet.yaml
#          trainer.devices=${devices}
#          mode=hip.yaml
#          +model.backbone.path_to_weights=${weights_40[$j]}
#          name=FT_sauvola_rlsa_new_3cl_csg18_polygon_unet16_loss_no_weights_40pt_100e
#          logger.wandb.project=hip
#          logger.wandb.tags=[unet16,csg18,polygon,3-classes,fine-tune,100-epochs,no-weights,3cl_new,rlsa_new_3cl,new_heuristic,40-epoch-pt,with_header]
#          logger.wandb.group=fine-tune-sauvola-rlsa-new-3cl-polygon-csg18-40pt"
#  python run.py ${params_18}
#
#  params_863="experiment=fine_tune_csg863_polygon_unet.yaml
#          trainer.devices=${devices}
#          mode=hip.yaml
#          +model.backbone.path_to_weights=${weights_40[$j]}
#          name=FT_sauvola_rlsa_new_3cl_csg863_polygon_unet16_loss_no_weights_40pt_100e
#          logger.wandb.project=hip
#          logger.wandb.tags=[unet16,csg863,polygon,3-classes,fine-tune,100-epochs,no-weights,3cl_new,rlsa_new_3cl,new_heuristic,40-epoch-pt,with_header]
#          logger.wandb.group=fine-tune-sauvola-rlsa-new-3cl-csg863-polygon-40pt"
#  python run.py ${params_863}
#  #    echo ${params}
#done

weights_50=("/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_heuristic_unet16_loss_no_weights_50epoch/2023-04-21/09-06-41/checkpoints/epoch\=45/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_heuristic_unet16_loss_no_weights_50epoch/2023-04-21/09-12-56/checkpoints/epoch\=26/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_heuristic_unet16_loss_no_weights_50epoch/2023-04-21/09-19-13/checkpoints/epoch\=34/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_heuristic_unet16_loss_no_weights_50epoch/2023-04-21/09-25-32/checkpoints/epoch\=29/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_heuristic_unet16_loss_no_weights_50epoch/2023-04-21/09-31-49/checkpoints/epoch\=30/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_heuristic_unet16_loss_no_weights_50epoch/2023-04-21/09-38-03/checkpoints/epoch\=48/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_heuristic_unet16_loss_no_weights_50epoch/2023-04-21/09-44-18/checkpoints/epoch\=27/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_heuristic_unet16_loss_no_weights_50epoch/2023-04-21/09-50-37/checkpoints/epoch\=44/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_heuristic_unet16_loss_no_weights_50epoch/2023-04-21/09-56-58/checkpoints/epoch\=23/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/PT_cb55_heuristic_unet16_loss_no_weights_50epoch/2023-04-21/10-03-12/checkpoints/epoch\=41/backbone.pth"
)

for j in "${!weights_50[@]}"; do
  params_18="experiment=fine_tune_csg18_polygon_unet16.yaml
          trainer.devices=${devices}
          mode=hip.yaml
          +model.backbone.layers_to_load=[conv1,conv2,conv3,conv4,bottleneck]
          +model.backbone.path_to_weights=${weights_50[$j]}
          name=FT_encoder_sauvola_rlsa_new_3cl_csg18_polygon_unet16_loss_no_weights_50pt_100e
          logger.wandb.project=hip
          logger.wandb.tags=[unet16,csg18,polygon,3-classes,fine-tune,100-epochs,no-weights,3cl_new,rlsa_new_3cl,new_heuristic,50-epoch-pt,with_header]
          logger.wandb.group=fine-tune-sauvola-rlsa-new-3cl-polygon-csg18-50pt"
  python run.py ${params_18}
#
#  params_863="experiment=fine_tune_csg863_polygon_unet.yaml
#          trainer.devices=${devices}
#          mode=hip.yaml
#          +model.backbone.path_to_weights=${weights_50[$j]}
#          name=FT_sauvola_rlsa_new_3cl_csg863_polygon_unet16_loss_no_weights_50pt_100e
#          logger.wandb.project=hip
#          logger.wandb.tags=[unet16,csg863,polygon,3-classes,fine-tune,100-epochs,no-weights,3cl_new,rlsa_new_3cl,new_heuristic,50-epoch-pt,with_header]
#          logger.wandb.group=fine-tune-sauvola-rlsa-new-3cl-csg863-polygon-50pt"
#  python run.py ${params_863}
  #    echo ${params}
done
