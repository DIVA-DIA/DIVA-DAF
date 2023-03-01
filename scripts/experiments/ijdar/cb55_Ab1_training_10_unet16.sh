#!/usr/bin/env bash

set -e

params_unet="experiment=cb55_AB1_train_20_run_unet16.yaml
        trainer.devices=[0,1]
        name=sem_seg_baseline_cb55_AB1_loss_no_weights_unet16_600ep
        logger.wandb.tags=[unet16,AB1,4-classes,baseline,600-epochs,no-weights,training-10,time-measure]
        datamodule.train_folder_name=training-10
        trainer.max_epochs=600"

python run.py ${params_unet}
