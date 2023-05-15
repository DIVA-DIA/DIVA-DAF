params="experiment=fine_tune_cb55_AB1_train_20_run_unet.yaml
          trainer.devices=[0,1]
          trainer.max_epochs=200
          mode=tiles_segmentation
          datamodule.train_folder_name=training-10
          ~model.header.path_to_weights
          name=baseline_cb55_AB1_training-10_unet_loss_no_weights_200ep
          logger.wandb.project=tiles_segmentation
          logger.wandb.tags=[unet,AB1,training-10,3-classes,baseline,200-epochs,no-weights]
          logger.wandb.group=baseline-cb55-3cl-training-10"
python run.py ${params}
