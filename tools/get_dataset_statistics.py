import json
from pathlib import Path

from PIL import Image
import numpy as np

if __name__ == '__main__':
    root_path = Path("/net/research-hisdoc/datasets/semantic_segmentation/datasets/CB55-splits/AB1_3class")
    test_gt_path = root_path / "test" / "gt"

    class_occurrences = []

    for img in test_gt_path.iterdir():
        img = Image.open(img)
        img_array = np.asarray(img)
        if len(class_occurrences) == 0:
            class_occurrences = np.bincount(img_array.flatten())
        else:
            class_occurrences += np.bincount(img_array.flatten())

    stats = {"pxl_per_class": class_occurrences.tolist(),
             "relative_per_class": (class_occurrences / class_occurrences.sum()).tolist(),
             "amount_of_pxls": class_occurrences.sum().tolist()}

    with (root_path / 'stats.json').open('w') as f:
        json.dump(stats, f)
