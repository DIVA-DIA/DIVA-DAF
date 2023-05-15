<div align="center">

# DIVA-DAF

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[![Coverage](https://sonarcloud.io/api/project_badges/measure?project=DIVA-DIA_DIVA-DAF&metric=coverage)](https://sonarcloud.io/summary/new_code?id=DIVA-DIA_DIVA-DAF)
![tests](https://github.com/DIVA-DIA/unsupervised_learning/actions/workflows/ci-testing.yml/badge.svg)
[![Maintainability Rating](https://sonarcloud.io/api/project_badges/measure?project=DIVA-DIA_DIVA-DAF&metric=sqale_rating)](https://sonarcloud.io/summary/new_code?id=DIVA-DIA_DIVA-DAF)

[comment]: <> ([![Paper]&#40;http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg&#41;]&#40;https://www.nature.com/articles/nature14539&#41;)

[comment]: <> ([![Conference]&#40;http://img.shields.io/badge/AnyConference-year-4b44ce.svg&#41;]&#40;https://papers.nips.cc/paper/2020&#41;)

</div>

## Description
A deep learning framework for historical document image analysis.

## How to run
Install dependencies
```yaml
# clone project
git clone https://github.com/DIVA-DIA/unsupervised_learning.git
cd unsupervised_learing

# create conda environment (IMPORTANT: needs Python 3.8+)
conda env create -f conda_env_gpu.yaml

# activate the environment using .autoenv
source .autoenv

# install requirements
pip install -r requirements.txt
```

Train model with default configuration.
Care: you need to change the value of `data_dir` in `config/datamodule/cb55_10_cropped_datamodule.yaml`.
```yaml
# default run based on config/config.yaml
python run.py

# train on CPU
python run.py trainer.gpus=0

# train on GPU
python run.py trainer.gpus=1
```

Train using GPU
```yaml
# [default] train on all available GPUs
python run.py trainer.gpus=-1

# train on one GPU
python run.py trainer.gpus=1

# train on two GPUs
python run.py trainer.gpus=2

# train on CPU
python run.py trainer.accelerator=ddp_cpu
```

Train using CPU for debugging
```yaml
# train on CPU
python run.py trainer.accelerator=ddp_cpu trainer.precision=32
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)
```yaml
python run.py experiment=experiment_name
```

You can override any parameter from command line like this
```yaml
python run.py trainer.max_epochs=20 datamodule.batch_size=64
```

<br>

### Setup PyCharm

1. Fork this repo
2. Clone the repo to your local filesystem (`git clone CLONELINK`)
3. Clone the repo onto your remote machine
4. Move into the folder on your remote machine and create the conda environment (conda env create -f conda_env_gpu.yaml)
5. Run `source .autoenv` in the root folder on your remote machine (activates the environment)
6. Open the folder in PyCharm (File -> open)
7. Add the interpreter (Preferences -> Project -> Python interpreter -> top left gear icon -> add... -> SSH Interpreter) follow the instructions (set the correct mapping to enable deployment)
8. Upload the files (deployment)
9. Create a wandb account (wandb.ai)
10. Log via ssh onto your remote machine 
11. Go to the root folder of the framework and activate the environment (source .autoenv OR conda activate unsupervised_learning)
12. Log into wandb. Execute `wandb login` and follow the instructions
13. Now you should be able to run the basic experiment from PyCharm


### Loading models
You can load the different model parts `backbone` or `header` as well as the whole task.
To load the `backbone` or the `header` you need to add to your experiment config the field `path_to_weights`.
e.g.
```
model:
    header:
        path_to_weights: /my/path/to/the/pth/file
```
To load the whole task you need to provide the path to the whole task to the trainer. This is with the field `resume_from_checkpoint`.
e.g.
```
trainer:
    resume_from_checkpoint: /path/to/.ckpt/file
```

### Freezing model parts
You can freeze both parts of the model (backbone or header) with the `freeze` flag in the config. 
E.g. you want to freeze the backbone:
In the command line:
```
python run.py +model.backbone.freeze=True
```
In the config (e.g. model/backbone/baby_unet.yaml):
```
...
freeze: True
...
```
CARE: You can not train a model when you do not have trainable parameters (e.g. freezing backbone and header).

### Selection in datasets
If you use the `selection` key you can either use an int, which takes the first n files, or a list of strings to filter the different datasets.
In the case you are using a full-page dataset be aware that the selection list is a list of file names without the extension.
    
    
### Cite us
```
@misc{vögtlin2022divadaf,
      title={DIVA-DAF: A Deep Learning Framework for Historical Document Image Analysis}, 
      author={Lars Vögtlin and Paul Maergner and Rolf Ingold},
      year={2022},
      eprint={2201.08295},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
