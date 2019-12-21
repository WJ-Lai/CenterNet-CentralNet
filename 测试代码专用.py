import torch.nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval

# GT
coco = coco.COCO('/media/vincent/856c2c04-3976-4948-ba47-5539ecaa24be/vincent/Code/CenterNet-cuda10-multi-spectral/data/rgb/annotations/test.json')
# Prediction
coco_dets = coco.loadRes('{}/results.json'.format('/media/vincent/856c2c04-3976-4948-ba47-5539ecaa24be/vincent/Checkpoint/CenterNet-CentralNet/ctdet/default/'))
coco_eval = COCOeval(coco, coco_dets, "bbox")
coco_eval.evaluate()
print('')
coco_eval.accumulate()
print('')
coco_eval.summarize()
# 返回mAP值
print(coco_eval.stats[1])