params="experiment=fine_tune_cb55_AB1_train_20_run_unet.yaml
          trainer.devices=[0,1]
          mode=tiles_segmentation
          datamodule.train_folder_name=training-10
          ~model.header.path_to_weights
          +model.backbone.path_to_weights=/net/research-hisdoc/experiments_lars_paul/lars_luca/tiles_segmentation/vinay_tiles_segmentation_unet/2023-03-16/18-42-02/checkpoints/epoch\=1/backbone.pth
          name=fine_tune_tiles_segmentation_cb55_AB1_training-10_unet_loss_no_weights_3pt_50e
          logger.wandb.project=tiles_segmentation
          logger.wandb.tags=[unet,AB1,training-10,3-classes,fine-tune,50-epochs,no-weights,gaussian,binary,3-epoch-pt]
          logger.wandb.group=fine-tune-tiles_segmentation-3pt-training-10"
python run.py ${params}
