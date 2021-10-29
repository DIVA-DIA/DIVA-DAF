from torch import nn

from src.models.utils.utils import Flatten


class SingleLinear(nn.Module):
    def __init__(self, num_classes: int = 4, input_size: int = 109512):
        super(SingleLinear, self).__init__()
        
        self.fc = nn.Sequential(
            Flatten(),
            nn.Linear(input_size, num_classes)
        )

    def forward(self, x):
        x = self.fc(x)
        return x