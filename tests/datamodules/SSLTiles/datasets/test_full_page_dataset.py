from src.datamodules.SSLTiles.datasets.full_page_dataset import DatasetSSLTiles
from tests.test_data.dummy_data_histdb_new.dummy_data import data_dir


def test_get_img_gt_path_list(data_dir):
    img_gt_path_list = DatasetSSLTiles.get_img_gt_path_list(directory=data_dir / 'train', data_folder_name='data',
                                                            gt_folder_name='gt',
                                                            selection=None)
    assert len(img_gt_path_list) == 2
    assert img_gt_path_list[0].name == 'fmb-cb-55-005v.png'
    assert img_gt_path_list[1].name == 'fmb-cb-55-005v_2.png'
