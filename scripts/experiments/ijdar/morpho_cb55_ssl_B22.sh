#!/usr/bin/env bash
set -e

params="experiment=morpho_cb55_B22_unet.yaml
    trainer.devices=[0,1,2,3]
    +trainer.precision=32
    mode=ijdar.yaml
    name=morpho_cb55_B22_unet_loss_no_weights-200ep-2
    trainer.max_epochs=200
    logger.wandb.project=ijdar_controlled
    logger.wandb.group=morpho-pt-cb55-unet
    logger.wandb.tags=[best_model,Morpho,pre-training,unet,B22-60,CB55-ssl-set,200-epochs]"
python run.py ${params}

params="experiment=morpho_cb55_B22_unet16.yaml
    trainer.devices=[0,1,2,3]
    +trainer.precision=32
    mode=ijdar.yaml
    name=morpho_cb55_B22_unet16_loss_no_weights-200ep
    trainer.max_epochs=200
    logger.wandb.project=ijdar_controlled
    logger.wandb.group=morpho-pt-cb55-unet
    logger.wandb.tags=[best_model,Morpho,pre-training,unet,B22-60,CB55-ssl-set,200-epochs]"
python run.py ${params}

params="experiment=morpho_cb55_B22_adaptive_unet.yaml
    trainer.devices=[0,1,2,3]
    +trainer.precision=32
    mode=ijdar.yaml
    name=morpho_cb55_B22_adaptive_unet_loss_no_weights-200ep
    trainer.max_epochs=200
    logger.wandb.project=ijdar_controlled
    logger.wandb.group=morpho-pt-cb55-unet
    logger.wandb.tags=[best_model,Morpho,pre-training,unet,B22-60,CB55-ssl-set,200-epochs]"
python run.py ${params}

params="experiment=morpho_cb55_B22_doc_ufcn.yaml
    trainer.devices=[0,1,2,3]
    +trainer.precision=32
    mode=ijdar.yaml
    name=morpho_cb55_B22_doc_ufcn_loss_no_weights-200ep
    trainer.max_epochs=200
    logger.wandb.project=ijdar_controlled
    logger.wandb.group=morpho-pt-cb55-unet
    logger.wandb.tags=[best_model,Morpho,pre-training,unet,B22-60,CB55-ssl-set,200-epochs]"
python run.py ${params}
