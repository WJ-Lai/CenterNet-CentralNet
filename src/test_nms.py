from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import json
import cv2
import numpy as np
import time
from progress.bar import Bar
import torch

from external.nms import soft_nms
from opts import opts
from logger import Logger
from utils.utils import AverageMeter
from datasets.dataset_factory import dataset_factory
from detectors.detector_factory import detector_factory
from external.nms import soft_nms

import copy

class PrefetchDataset(torch.utils.data.Dataset):
  def __init__(self, opt, dataset, pre_process_func):
    self.images = dataset.images
    self.load_image_func = dataset.coco.loadImgs
    self.img_dir = dataset.img_dir
    self.pre_process_func = pre_process_func
    self.opt = opt
  
  def __getitem__(self, index):
    img_id = self.images[index]
    img_info = self.load_image_func(ids=[img_id])[0]
    img_path = os.path.join(self.img_dir, img_info['file_name'])
    image = cv2.imread(img_path)
    images, meta = {}, {}
    for scale in opt.test_scales:
      if opt.task == 'ddd':
        images[scale], meta[scale] = self.pre_process_func(
          image, scale, img_info['calib'])
      else:
        images[scale], meta[scale] = self.pre_process_func(image, scale)
    return img_id, {'images': images, 'image': image, 'meta': meta}

  def __len__(self):
    return len(self.images)

def prefetch_test(opt):
  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str

  Dataset = dataset_factory[opt.dataset]
  opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
  print(opt)
  Logger(opt)
  Detector = detector_factory[opt.task]
  
  split = 'val' if not opt.trainval else 'test'
  dataset = Dataset(opt, split)
  detector = Detector(opt)
  
  data_loader = torch.utils.data.DataLoader(
    PrefetchDataset(opt, dataset, detector.pre_process), 
    batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

  results = {}
  num_iters = len(dataset)
  bar = Bar('{}'.format(opt.exp_id), max=num_iters)
  time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']
  avg_time_stats = {t: AverageMeter() for t in time_stats}
  for ind, (img_id, pre_processed_images) in enumerate(data_loader):
    ret = detector.run(pre_processed_images)
    results[img_id.numpy().astype(np.int32)[0]] = ret['results']
    Bar.suffix = '[{0}/{1}]|Tot: {total:} |ETA: {eta:} '.format(
                   ind, num_iters, total=bar.elapsed_td, eta=bar.eta_td)
    for t in avg_time_stats:
      avg_time_stats[t].update(ret[t])
      Bar.suffix = Bar.suffix + '|{} {tm.val:.3f}s ({tm.avg:.3f}s) '.format(
        t, tm = avg_time_stats[t])
    bar.next()
  bar.finish()
  dataset.run_eval(results, opt.save_dir)

def test(opt):
  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str

  Dataset = dataset_factory[opt.dataset]
  opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
  print(opt)
  Logger(opt)
  Detector = detector_factory[opt.task]
  
  split = 'val' if not opt.trainval else 'test'
  dataset = Dataset(opt, split)
  detector = Detector(opt)

  results = {}
  num_iters = len(dataset)
  bar = Bar('{}'.format(opt.exp_id), max=num_iters)
  time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']
  avg_time_stats = {t: AverageMeter() for t in time_stats}
  for ind in range(num_iters):
    img_id = dataset.images[ind]
    img_info = dataset.coco.loadImgs(ids=[img_id])[0]
    img_path = os.path.join(dataset.img_dir, img_info['file_name'])

    if opt.task == 'ddd':
      ret = detector.run(img_path, img_info['calib'])
    else:
      ret = detector.run(img_path)
    
    results[img_id] = ret['results']

  return results, dataset

def cal_bbox(results):
  bbox_sum = 0
  for _, img in results.items():
    for _, categories_id in img.items():
      bbox_num,_ = np.shape(categories_id)
      bbox_sum += bbox_num
  return bbox_sum

def nms(results1, results2):
  bbox_change = 0
  for img_id in results1.keys():
    for categories_id in range(1, 5):
      results1[img_id][categories_id] = np.vstack((results1[img_id][categories_id],results2[img_id][categories_id]))
      bbox_num_before = cal_bbox(results1)
      soft_nms(results1[img_id][categories_id], Nt=0.5, method=2)
      bbox_num_after = cal_bbox(results1)
      bbox_change += bbox_num_before-bbox_num_after
  print('Bounding Box change: %s' % bbox_change)
  return results1

def cal_mAP(results, dataset):
  dataset.run_eval(results, opt.save_dir)

def save_result(results_rgb, results_fir, results_mir, results_nir):
  np.save('results_rgb.npy', results_rgb)
  np.save('results_fir.npy', results_fir)
  np.save('results_mir.npy', results_mir)
  np.save('results_nir.npy', results_nir)

def load_result():
  results_rgb = np.load('results_rgb.npy').item()
  results_fir = np.load('results_fir.npy').item()
  results_mir = np.load('results_mir.npy').item()
  results_nir = np.load('results_nir.npy').item()
  Dataset = dataset_factory['rgb']
  dataset = Dataset(opt, 'test')
  return results_rgb, results_fir, results_mir, results_nir, dataset

def output_result(opt, dataset):
  opt.dataset = dataset
  opt.load_model = '/home/vincent/Checkpoint/CenterNet-CentralNet/ctdet/single2/'+dataset+'/model_last.pth'
  results, dataset = test(opt)
  return results, dataset

def output_fusion_mAP():
  # results_rgb, results_fir, results_mir, results_nir, dataset = load_result()
  results_sensor = load_result()
  for i in range(1,4):
    results_rgb = copy.deepcopy(results_sensor[0])
    results = nms(results_rgb, results_sensor[i])
    cal_mAP(results, results_sensor[4])

def output_fusion_mAP_rgb_after():
  # results_rgb, results_fir, results_mir, results_nir, dataset = load_result()
  results_sensor = load_result()
  for i in range(1, 4):
    results = nms(results_sensor[i], results_sensor[0])
    cal_mAP(results, results_sensor[4])

def output_sing_mAP():
  results_rgb, results_fir, results_mir, results_nir, dataset = load_result()
  cal_mAP(results_rgb, dataset)
  cal_mAP(results_fir, dataset)
  cal_mAP(results_mir, dataset)
  cal_mAP(results_nir, dataset)

if __name__ == '__main__':
  opt = opts().parse()
  # results_rgb, dataset = output_result(opt, 'rgb')
  # results_fir, dataset = output_result(opt, 'fir')
  # results_mir, dataset = output_result(opt, 'mir')
  # results_nir, dataset = output_result(opt, 'nir')
  # save_result(results_rgb, results_fir, results_mir, results_nir)

  # output_sing_mAP()
  output_fusion_mAP()
  # output_fusion_mAP_rgb_after()