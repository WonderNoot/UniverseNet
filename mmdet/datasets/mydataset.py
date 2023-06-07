import mmcv

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class MyDataset(CustomDataset):

    CLASSES = ('ILS_vertical_filled', 'ILS_horizontal_filled', 'red_light', 'yellow_light', 'ILS_horizontal_empty', 'ILS_vertical_empty')

    def load_annotations(self, ann_file):
        ann_list = mmcv.list_from_file(ann_file)