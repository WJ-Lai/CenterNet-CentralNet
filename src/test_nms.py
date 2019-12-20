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
from itertools import combinations
import copy

import sys


class Logger_txt(object):
  def __init__(self, fileN="Default.log"):
    self.terminal = sys.stdout
    self.log = open(fileN, "a")

  def write(self, message):
    self.terminal.write(message)
    self.log.write(message)

  def flush(self):
    pass

def test(opt):
  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str

  Dataset = dataset_factory[opt.dataset]
  opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
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

def nms(*args):
  results = []
  for sensor in args:
    results.append(sensor)
  bbox_change = 0
  for img_id in results[0].keys():
    for categories_id in range(1, 5):
      res_stack = [results[i][img_id][categories_id] for i in range(len(results))]
      results[0][img_id][categories_id] = np.vstack(res_stack)
      bbox_num_before = cal_bbox(results[0])
      soft_nms(results[0][img_id][categories_id], Nt=0.5, method=2)
      bbox_num_after = cal_bbox(results[0])
      bbox_change += bbox_num_before-bbox_num_after
  print('Bounding Box change: %s' % bbox_change)
  return results[0]

def nms_multi(results):
  bbox_change = 0
  for img_id in results[0].keys():
    for categories_id in range(1, 5):
      res_stack = [results[i][img_id][categories_id] for i in range(len(results))]
      results[0][img_id][categories_id] = np.vstack(res_stack)
      bbox_num_before = cal_bbox(results[0])
      soft_nms(results[0][img_id][categories_id], Nt=0.5, method=2)
      bbox_num_after = cal_bbox(results[0])
      bbox_change += bbox_num_before-bbox_num_after
  print('Bounding Box change: %s' % bbox_change)
  return results[0]

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

def output_result(opt, dataset, model_type):
  assert model_type in [333,666,'best','last'], 'Model type must in [333,666,\'best\',\'last\']'
  opt.dataset = dataset
  opt.load_model = '/media/vincent/856c2c04-3976-4948-ba47-5539ecaa24be/vincent/Checkpoint/CenterNet-CentralNet/ctdet/single2/'+dataset+'/model_'+str(model_type)+'.pth'
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

def get_all_sensor_combinations():
  sensor = ['rgb', 'fir', 'mir', 'nir']
  tmp_list, sensor_combinations = [], []
  for i in range(1, len(sensor)+1):
      iter1 = combinations(sensor, i)
      tmp_list.append(iter1)

  for i in tmp_list:
      for j in i:
          sensor_combinations.append(list(j))
  return sensor_combinations


def output_all_results(model_type):
  print(model_type)
  results_rgb, dataset = output_result(opt, 'rgb', model_type)
  results_fir, dataset = output_result(opt, 'fir', model_type)
  results_mir, dataset = output_result(opt, 'mir', model_type)
  results_nir, dataset = output_result(opt, 'nir', model_type)
  save_result(results_rgb, results_fir, results_mir, results_nir)

  # output_sing_mAP()
  # output_fusion_mAP()
  # output_fusion_mAP_rgb_after()

  sensor_combinations = get_all_sensor_combinations()
  results_rgb, results_fir, results_mir, results_nir, dataset = load_result()
  results_dict = {'rgb': results_rgb, 'fir': results_fir, 'mir': results_mir, 'nir': results_nir}
  for sensor_list in sensor_combinations:
    results = []
    for sensor in sensor_list:
      results.append(copy.deepcopy(results_dict[sensor]))
    results = nms_multi(results)
    print(sensor_list)
    cal_mAP(results, dataset)


if __name__ == '__main__':
  sys.stdout = Logger_txt("results.txt")
  opt = opts().parse()
  for opt.test_dataset in ['test', 'val']:
    print(opt.test_dataset)
    for model_type in [333,666,'best','last']:
      output_all_results(model_type)
      print('')
      print('')
    print('')
    print('')
