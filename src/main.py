from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os

import torch
import torch.utils.data
from opts import opts
from models.model import create_model, load_model, save_model
from models.data_parallel import DataParallel
from logger import Logger
from datasets.dataset_factory import get_dataset
from trains.train_factory import train_factory
from progress.bar import Bar
from datasets.dataset_factory import dataset_factory
from detectors.detector_factory import detector_factory
from utils.utils import AverageMeter

def test(opt, dataset):
    opt.load_model = os.path.join(opt.save_dir, 'model_last.pth')
    Detector = detector_factory[opt.task]
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

        Bar.suffix = '[{0}/{1}]|Tot: {total:} |ETA: {eta:} '.format(
            ind, num_iters, total=bar.elapsed_td, eta=bar.eta_td)
        for t in avg_time_stats:
            avg_time_stats[t].update(ret[t])
            Bar.suffix = Bar.suffix + '|{} {:.3f} '.format(t, avg_time_stats[t].avg)
        bar.next()
    bar.finish()
    return dataset.cal_mAP(results, opt.save_dir)

def main(opt):

  torch.manual_seed(opt.seed)
  torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test
  Dataset = get_dataset(opt.dataset[0], opt.task)
  opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
  print(opt)

  logger = Logger(opt)

  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')
  
  print('Creating model...')
  model = create_model(opt.arch, opt.heads, opt.head_conv)
  # print(model)

  print(model._name)
  params = list(model.parameters())
  k = 0
  for i in params:
      l = 1
      # print("该层的结构：" + str(list(i.size())))
      for j in i.size():
          l *= j
      # print("该层参数和：" + str(l))
      k = k + l
  print("总参数数量和：" + str(k))


  # optimizer = torch.optim.Adam(model.parameters(), opt.lr, weight_decay=opt.l2_norm)
  optimizer = torch.optim.Adam(model.parameters(), opt.lr)

  start_epoch = 0
  if opt.load_model != '':
    model, optimizer, start_epoch = load_model(
      model, opt.load_model, optimizer, opt.resume, opt.lr, opt.lr_step)

  Trainer = train_factory[opt.task]
  trainer = Trainer(opt, model, optimizer)
  trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)

  print('Setting up data...')
  val_loader = torch.utils.data.DataLoader(
      Dataset(opt, 'val'), 
      batch_size=1, 
      shuffle=False,
      num_workers=opt.num_workers,
      pin_memory=True
  )

  if opt.test:
    _, preds = trainer.val(0, val_loader)
    val_loader.dataset.run_eval(preds, opt.save_dir)
    return

  train_loader = torch.utils.data.DataLoader(
      Dataset(opt, 'train'), 
      batch_size=opt.batch_size, 
      shuffle=True,
      num_workers=opt.num_workers,
      pin_memory=True,
      drop_last=True
  )

  print('Starting training...')
  if opt.load_model:
      if opt.metric == 'loss':
        best = 1e10
      else:
        best = test(opt, Dataset(opt, 'val'))
  else:
      if opt.metric == 'loss':
        best = 1e10
      else:
        best = 0.0


  for epoch in range(start_epoch + 1, opt.num_epochs + 1):
    mark = epoch if opt.save_all else 'last'
    log_dict_train, _ = trainer.train(epoch, train_loader)
    logger.write('epoch: {} |'.format(epoch))
    for k, v in log_dict_train.items():
      logger.scalar_summary('train_{}'.format(k), v, epoch)
      logger.write('{} {:8f} | '.format(k, v))
    if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
      save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(mark)), 
                 epoch, model, optimizer)
      with torch.no_grad():
        if opt.metric=='loss':
          log_dict_val, preds = trainer.val(epoch, val_loader)
        else:
          log_dict_val = {}
          train_mAP = {}
          log_dict_val[opt.metric] = test(opt, Dataset(opt, 'val'))
          train_mAP[opt.metric] = test(opt, Dataset(opt, 'train'))
          print('Train mAP:{}'.format(train_mAP[opt.metric]))
          print('Val mAP:{}'.format(log_dict_val[opt.metric]))
      for k, v in log_dict_val.items():
        logger.scalar_summary('val_{}'.format(k), v, epoch)
        logger.write('{} {:8f} | '.format(k, v))
      for k, v in train_mAP.items():
        logger.scalar_summary('train_{}'.format(k), v, epoch)
        logger.write('{} {:8f} | '.format(k, v))
      if opt.metric=='loss':
          if log_dict_val[opt.metric] < best:
            best = log_dict_val[opt.metric]
            save_model(os.path.join(opt.save_dir, 'model_best.pth'),
                       epoch, model)
      else:
          if log_dict_val[opt.metric] > best:
            best = log_dict_val[opt.metric]
            save_model(os.path.join(opt.save_dir, 'model_best.pth'),
                       epoch, model)
            print('save best model of epoch {}, mAP={}'.format(epoch, best))
    else:
      save_model(os.path.join(opt.save_dir, 'model_last.pth'), 
                 epoch, model, optimizer)
    logger.write('\n')
    if epoch in opt.lr_step:
      save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)), 
                 epoch, model, optimizer)
      lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
      print('Drop LR to', lr)
      for param_group in optimizer.param_groups:
          param_group['lr'] = lr
  logger.close()


if __name__ == '__main__':
  opt = opts().parse()
  # opt.dataset = 'rgb'
  # opt.arch = 'hourglass'
  main(opt)