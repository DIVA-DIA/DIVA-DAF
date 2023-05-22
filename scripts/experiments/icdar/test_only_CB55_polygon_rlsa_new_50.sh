#!/usr/bin/env bash

set -e

weights_bb=("/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/3cl_new_rlsa_cb55_sauvola_unet_loss_no_weights_50epoch/2023-02-07/19-43-21/checkpoints/epoch\=21/backbone.pth"
"/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/3cl_new_rlsa_cb55_sauvola_unet_loss_no_weights_50epoch/2023-02-07/19-54-52/checkpoints/epoch\=34/backbone.pth"
"/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/3cl_new_rlsa_cb55_sauvola_unet_loss_no_weights_50epoch/2023-02-07/20-06-20/checkpoints/epoch\=20/backbone.pth"
"/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/3cl_new_rlsa_cb55_sauvola_unet_loss_no_weights_50epoch/2023-02-07/20-17-51/checkpoints/epoch\=14/backbone.pth"
"/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/3cl_new_rlsa_cb55_sauvola_unet_loss_no_weights_50epoch/2023-02-07/20-29-21/checkpoints/epoch\=17/backbone.pth"
"/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/3cl_new_rlsa_cb55_sauvola_unet_loss_no_weights_50epoch/2023-02-07/20-40-54/checkpoints/epoch\=44/backbone.pth"
"/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/3cl_new_rlsa_cb55_sauvola_unet_loss_no_weights_50epoch/2023-02-07/20-52-35/checkpoints/epoch\=26/backbone.pth"
"/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/3cl_new_rlsa_cb55_sauvola_unet_loss_no_weights_50epoch/2023-02-07/21-04-12/checkpoints/epoch\=15/backbone.pth"
"/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/3cl_new_rlsa_cb55_sauvola_unet_loss_no_weights_50epoch/2023-02-07/21-15-44/checkpoints/epoch\=27/backbone.pth"
"/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/3cl_new_rlsa_cb55_sauvola_unet_loss_no_weights_50epoch/2023-02-07/21-27-20/checkpoints/epoch\=46/backbone.pth")

weights_header=("/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/3cl_new_rlsa_cb55_sauvola_unet_loss_no_weights_50epoch/2023-02-07/19-43-21/checkpoints/epoch\=21/header.pth"
"/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/3cl_new_rlsa_cb55_sauvola_unet_loss_no_weights_50epoch/2023-02-07/19-54-52/checkpoints/epoch\=34/header.pth"
"/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/3cl_new_rlsa_cb55_sauvola_unet_loss_no_weights_50epoch/2023-02-07/20-06-20/checkpoints/epoch\=20/header.pth"
"/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/3cl_new_rlsa_cb55_sauvola_unet_loss_no_weights_50epoch/2023-02-07/20-17-51/checkpoints/epoch\=14/header.pth"
"/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/3cl_new_rlsa_cb55_sauvola_unet_loss_no_weights_50epoch/2023-02-07/20-29-21/checkpoints/epoch\=17/header.pth"
"/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/3cl_new_rlsa_cb55_sauvola_unet_loss_no_weights_50epoch/2023-02-07/20-40-54/checkpoints/epoch\=44/header.pth"
"/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/3cl_new_rlsa_cb55_sauvola_unet_loss_no_weights_50epoch/2023-02-07/20-52-35/checkpoints/epoch\=26/header.pth"
"/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/3cl_new_rlsa_cb55_sauvola_unet_loss_no_weights_50epoch/2023-02-07/21-04-12/checkpoints/epoch\=15/header.pth"
"/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/3cl_new_rlsa_cb55_sauvola_unet_loss_no_weights_50epoch/2023-02-07/21-15-44/checkpoints/epoch\=27/header.pth"
"/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/3cl_new_rlsa_cb55_sauvola_unet_loss_no_weights_50epoch/2023-02-07/21-27-20/checkpoints/epoch\=46/header.pth")


for j in "${!weights_bb[@]}"; do
  params="experiment=test_cb55_run_unet.yaml
        trainer.devices=[4]
        +model.backbone.path_to_weights=${weights_bb[$j]}
        ++model.header.path_to_weights=${weights_header[$j]}
        datamodule.data_dir=/net/research-hisdoc/datasets/semantic_segmentation/datasets/polygon_gt/CSG863
        name=test_csg863_polygon_unet_loss_no_weights_w_header_50pt
        logger.wandb.tags=[unet,CSG863,testing,3-classes,polygon]
        logger.wandb.group=test-polygon"
  python run.py ${params}
#    echo ${params}
done
