.

Installation
============

The installation requires anaconda or miniconda to be installed. If you don't have it installed, you can download it from [here](https://www.anaconda.com/distribution/).
Teh conda environment is just for linux. For other operating systems, you can use the requirements.txt file to install the required packages.

.. code-block:: bash

    # clone project
    git clone https://github.com/DIVA-DIA/unsupervised_learning.git
    cd unsupervised_learing

    # create conda environment (IMPORTANT: needs Python 3.8+)
    conda env create -f conda_env_gpu.yaml

    # activate the environment using .autoenv
    source .autoenv

    # install requirements
    pip install -r requirements.txt

To run the tests you need to also install the `tests/requirements.txt` file. The tests are run using `pytest`.