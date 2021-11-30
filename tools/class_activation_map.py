import importlib
import inspect
from pathlib import Path
from typing import List, Any

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image

from src.datamodules.utils.dataset_predict import DatasetPredict
from src.datamodules.utils.misc import ImageDimensions
from src.datamodules.utils.wrapper_transforms import OnlyImage


def _get_class(model_tuple_list: List[Any], model_name: str, weights: Path, num_classes: int = None):
    for n, c in model_tuple_list:
        if n == model_name:
            if num_classes is not None:
                m = c(num_classes=num_classes)
            else:
                m = c()
            m.load_state_dict(torch.load(weights), strict=True)
            return m
    return None


if __name__ == '__main__':
    import_raw = importlib.import_module('src.models.backbones')
    model_classes = inspect.getmembers(import_raw, inspect.isclass)

    weight_path = Path(
        "/netscratch/experiments_lars_paul/lars/experiments/semantic_segmentation_cb55_full_unet/2021-11-29/18-04-27/"
        "checkpoints/epoch=15/backbone.pth")

    model = _get_class(model_tuple_list=model_classes, model_name='UNet', weights=weight_path, num_classes=8)
    layer = list(model.modules())[94]

    mean = [0.7425630710515175, 0.6906955927303446, 0.5948711113866068]
    std = [0.3278947164057723, 0.31449158276334094, 0.2870113305648434]

    img_path_list = [
        '/netscratch/datasets/semantic_segmentation/datasets_cropped/CB55/test/data/e-codices_fmb-cb-0055_0098v_max/'
        'e-codices_fmb-cb-0055_0098v_max_x2432_y3584.png']

    img_np = np.array(Image.open(img_path_list[0]))
    img_tensor = preprocess_image(img_np, mean=mean, std=std)

    cam = GradCAM(model=model, target_layers=[layer])
    target_class = 1
    grayscale_cam = cam(input_tensor=img_tensor, target_category=target_class)

    grayscale_cam = grayscale_cam[0, :]
    viz = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
