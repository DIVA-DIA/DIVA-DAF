from pytorch_lightning.utilities import LightningEnum


class OutputKeys(LightningEnum):

    PREDICTION = 'pred'
    TARGET = 'target'
    LOG = 'logs'
    LOSS = 'loss'
