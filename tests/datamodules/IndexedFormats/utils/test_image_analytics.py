import numpy as np

from src.datamodules.IndexedFormats.datasets.full_page_dataset import DatasetIndexed
from src.datamodules.IndexedFormats.utils.image_analytics import get_analytics, \
    _get_class_frequencies_weights_segmentation_indexed
from tests.test_data.dummy_fixed_gif.dummy_data import data_dir

CLASS_ENCODINGS = np.asarray([[0, 0, 0], [0, 255, 255], [255, 0, 255], [255, 255, 0]])
CLASS_WEIGHTS = np.asarray([4.599257495869867e-07, 5.178234843306613e-06, 0.00011135857461024499, 4.899078973153047e-06])


def test_get_analytics(data_dir):
    analytics_data, analytics_gt = get_analytics(input_path=data_dir, data_folder_name='data', gt_folder_name='gt',
                                                 train_folder_name='train',
                                                 get_img_gt_path_list_func=DatasetIndexed.get_img_gt_path_list)
    assert np.allclose(analytics_data['mean'], [0.8267679793271475, 0.7056573666793107, 0.600163116027661], rtol=2e-03)
    assert np.allclose(analytics_data['std'], [0.26603996365851545, 0.24160881943877677, 0.21738707405192137], rtol=2e-03)
    assert analytics_data['width'] == 960
    assert analytics_data['height'] == 1344
    assert np.allclose(analytics_gt['class_weights'], CLASS_WEIGHTS)
    assert np.array_equal(analytics_gt['class_encodings'], CLASS_ENCODINGS)


def test__get_class_frequencies_weights_segmentation_indexed(data_dir):
    img_gt_path_list = list((data_dir / 'train' / 'gt').iterdir())
    file_names_gt = np.asarray(img_gt_path_list)
    class_weights, class_encodings = _get_class_frequencies_weights_segmentation_indexed(gt_images=file_names_gt)

    assert np.array_equal(class_encodings, CLASS_ENCODINGS)
    assert np.allclose(class_weights, CLASS_WEIGHTS)
