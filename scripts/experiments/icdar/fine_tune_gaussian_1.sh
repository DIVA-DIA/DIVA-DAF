params="experiment=fine_tune_cb55_AB1_train_20_run_unet.yaml
          trainer.devices=[4,5,6,7]
          datamodule.train_folder_name=training-20
          +model.backbone.path_to_weights=/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_gaussian_unet_loss_no_weights_20epochs/2023-01-17/15-19-24/checkpoints/epoch\=18/backbone.pth
          name=fine_tune_gaussian_cb55_AB1_training-20_unet_loss_no_weights_20pt_100e
          logger.wandb.tags=[unet,AB1,training-20,3-classes,fine-tune,100-epochs,no-weights,gaussian,binary,20-epoch-pt]
          logger.wandb.group=fine-tune-gaussian-20pt-training-20"
python run.py ${params}
