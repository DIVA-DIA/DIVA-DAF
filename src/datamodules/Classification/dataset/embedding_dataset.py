import json
from pathlib import Path
from typing import Tuple, Any, Callable

import torch
from torch.utils.data import Dataset


class EmbeddingsDataset(Dataset):

    def __init__(
            self,
            root: Path,
            split: str = "train",
            embedding: str = "aws_text-embedding-3-small"
    ):
        super().__init__()
        self.dataset_path = root
        self.split = split
        self.embedding = embedding

        self.data = list(
            filter(lambda s: self.split in str(s) and self.embedding in str(s), self.dataset_path.iterdir()))
        self.embedding_size = json.load(self.data[0].open('r'))['embedding_size']

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        json_content = json.load(self.data[idx].open('r'))
        embedding = json_content['embedding']
        target = json_content['id'].split('_')[-1]

        return torch.Tensor(embedding), int(target)
