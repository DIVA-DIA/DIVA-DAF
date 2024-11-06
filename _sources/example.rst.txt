
Example
=======

We are training a small U-Net model on the DIVAHisDB dataset.
Train model with default configuration.
Care: you need to change the value of data_dir in config/datamodule/cb55_10_cropped_datamodule.yaml.

.. code-block:: bash

    # default run based on config/config.yaml
    python run.py

    # train on CPU
    python run.py trainer.gpus=0

    # train on GPU
    python run.py trainer.gpus=1
    Train using GPU

    # [default] train on all available GPUs
    python run.py trainer.gpus=-1

    # train on one GPU
    python run.py trainer.gpus=1

    # train on two GPUs
    python run.py trainer.gpus=2

    # train on CPU
    python run.py trainer.accelerator=ddp_cpu