#!/usr/bin/env bash

set -e

weight="/net/research-hisdoc/experiments_lars_paul/lars_luca/ijdar/morpho_cb55_B22_unet_loss_no_weights-200ep-2/2023-03-12/09-12-43/checkpoints/epoch\=47/backbone.pth"

for j in {0..4}; do
  params_unet="experiment=fine_tune_cb55_AB1_train_20_run_unet.yaml
        trainer.devices=[0]
        +trainer.precision=32
        trainer.max_epochs=100
        mode=ijdar.yaml
        name=fine_tune_x5_morpho_cb55_AB1_training-10_run_unet-100ep
        +model.backbone.path_to_weights=${weight}
        datamodule.train_folder_name=training-10
        logger.wandb.tags=[unet,AB1,training-10,fine-tune,morpho-x5,4-classes,baseline,100-epochs,no-weights,best-jaccard]
        logger.wandb.project=ijdar_controlled
        logger.wandb.group=x5-morpho-FT-training-10-100ep"
  #    echo ${params_unet}
  python run.py ${params_unet}
done

weight="/net/research-hisdoc/experiments_lars_paul/lars_luca/ijdar/morpho_cb55_B22_unet16_loss_no_weights-200ep/2023-03-12/10-00-02/checkpoints/epoch\=152/backbone.pth"

for j in {0..4}; do
  params_unet="experiment=fine_tune_cb55_AB1_train_20_run_unet16.yaml
        trainer.devices=[0]
        +trainer.precision=32
        trainer.max_epochs=100
        mode=ijdar.yaml
        name=fine_tune_x5_morpho_cb55_AB1_training-10_run_unet16-100ep
        +model.backbone.path_to_weights=${weight}
        datamodule.train_folder_name=training-10
        logger.wandb.tags=[unet16,AB1,training-10,fine-tune,morpho-cb,4-classes,baseline,100-epochs,no-weights,best-jaccard]
        logger.wandb.project=ijdar_controlled
        logger.wandb.group=x5-morpho-cb-FT-training-10-100ep"
  #    echo ${params_unet}
  python run.py ${params_unet}
done

weight="/net/research-hisdoc/experiments_lars_paul/lars_luca/ijdar/morpho_cb55_B22_adaptive_unet_loss_no_weights-200ep/2023-03-12/10-13-36/checkpoints/epoch\=67/backbone.pth"

for j in {0..4}; do
  params_unet="experiment=fine_tune_cb55_AB1_train_20_run_adaptive_unet.yaml
        trainer.devices=[0]
        +trainer.precision=32
        trainer.max_epochs=100
        mode=ijdar.yaml
        name=fine_tune_x5_morpho_cb55_AB1_training-10_run_adaptive_unet-100ep
        +model.backbone.path_to_weights=${weight}
        datamodule.train_folder_name=training-10
        logger.wandb.tags=[unet_adaptive,AB1,training-10,fine-tune,morpho-x5,4-classes,baseline,100-epochs,no-weights,best-jaccard]
        logger.wandb.project=ijdar_controlled
        logger.wandb.group=x5-morpho-FT-training-10-100ep"
  #    echo ${params_unet}
  python run.py ${params_unet}
done

weight="/net/research-hisdoc/experiments_lars_paul/lars_luca/ijdar/morpho_cb55_B22_doc_ufcn_loss_no_weights-200ep/2023-03-12/10-33-23/checkpoints/epoch\=178/backbone.pth"

for j in {0..4}; do
  params_unet="experiment=fine_tune_cb55_AB1_train_20_run_doc_ufcn.yaml
        trainer.devices=[0]
        +trainer.precision=32
        trainer.max_epochs=100
        mode=ijdar.yaml
        name=fine_tune_x5_morpho_cb55_AB1_training-10_run_doc_ufcn-100ep
        +model.backbone.path_to_weights=${weight}
        datamodule.train_folder_name=training-10
        logger.wandb.tags=[doc_ufcn,AB1,training-10,fine-tune,morpho-x5,4-classes,baseline,100-epochs,no-weights,best-jaccard]
        logger.wandb.project=ijdar_controlled
        logger.wandb.group=x5-morpho-FT-training-10-100ep"
  #    echo ${params_unet}
  python run.py ${params_unet}
done
