import torch.nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval

coco = coco.COCO('J:/Python program/CenterNet_2/data/BDD/annotations/test.json')
coco_dets = coco.loadRes('{}/results.json'.format('J:/Python program/CenterNet_2/exp/ctdet/default/'))
coco_eval = COCOeval(coco, coco_dets, "bbox")
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()