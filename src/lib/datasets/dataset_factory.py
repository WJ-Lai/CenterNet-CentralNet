from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .sample.ddd import DddDataset
from .sample.exdet import EXDetDataset
from .sample.ctdet import CTDetDataset
from .sample.multi_pose import MultiPoseDataset
from .sample.ctdet_fusion import CTDetDataset_fusion

from .dataset.coco import COCO
from .dataset.pascal import PascalVOC
from .dataset.kitti import KITTI
from .dataset.coco_hp import COCOHP
from .dataset.voc import VOC
from .dataset.rgb import RGB
from .dataset.fir import FIR
from .dataset.mir import MIR
from .dataset.nir import NIR
from .dataset.camera import CAMERA


dataset_factory = {
  'coco': COCO,
  'pascal': PascalVOC,
  'kitti': KITTI,
  'coco_hp': COCOHP,
  'voc': VOC,
  'rgb': RGB,
  'fir': FIR,
  'mir': MIR,
  'nir': NIR,
  'fusion':CAMERA
}

_sample_factory = {
  'exdet': EXDetDataset,
  'ctdet': CTDetDataset,
  'ddd': DddDataset,
  'multi_pose': MultiPoseDataset,
  'ctdet_fusion': CTDetDataset_fusion
}


def get_dataset(dataset, task):
  class Dataset(dataset_factory[dataset], _sample_factory[task]):
    print('################## Dataset about %s ##################' % dataset)
    pass
  return Dataset
  
