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

# from tensorboardX import SummaryWriter


def print_weight(model, doc):
    layer_name_list = ['mergeup_pre', 'mergeup_kp', 'mergeup_head']
    for layer_name in layer_name_list:
        print(layer_name, file=doc)
        if layer_name in ['mergeup_kp', 'mergeup_head']:
            for i in range(len(model._modules[layer_name]._modules)):
                print(layer_name+str(i), file=doc)
                for j in range(len(model._modules[layer_name]._modules[str(i)].alpha)):
                    print(model._modules[layer_name]._modules[str(i)].alpha[j].data, file=doc)
        else:
            for i in range(len(model._modules[layer_name].alpha)):
                print(model._modules[layer_name].alpha[i].data, file=doc)
        print('', file=doc)

# class ConcatDataset(torch.utils.data.Dataset):
#     def __init__(self, *datasets):
#         self.datasets = datasets
#
#     def __getitem__(self, i):
#         return tuple(d[i] for d in self.datasets)
#
#     def __len__(self):
#         return min(len(d) for d in self.datasets)

def main(opt):

  torch.manual_seed(opt.seed)
  torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test

  Dataset = get_dataset('fusion', 'ctdet_fusion')
  opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
  print(opt)

  logger = Logger(opt)

  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  opt.device = torch.device('cuda:'+opt.gpus_str if opt.gpus[0] >= 0 else 'cpu')

  print('Creating model...')
  model = create_model('hourglassfusionnms', opt.heads, opt.head_conv)
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
  best = 1e10


  for epoch in range(start_epoch + 1, opt.num_epochs + 1):
    # with open(opt.save_dir+'\\'+opt.exp_id+'-weight.txt', 'w+') as doc:
    #     print_weight(model, doc)
    mark = epoch if opt.save_all else 'last'
    log_dict_train, _ = trainer.train_fusion_loader(epoch, train_loader)
    logger.write('epoch: {} |'.format(epoch))
    for k, v in log_dict_train.items():
      logger.scalar_summary('train_{}'.format(k), v, epoch)
      logger.write('{} {:8f} | '.format(k, v))
    if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
      save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(mark)), 
                 epoch, model, optimizer)
      with torch.no_grad():
        log_dict_val, preds = trainer.val_fusion_loader(epoch, val_loader)
      for k, v in log_dict_val.items():
        logger.scalar_summary('val_{}'.format(k), v, epoch)
        logger.write('{} {:8f} | '.format(k, v))
      if log_dict_val[opt.metric] < best:
        best = log_dict_val[opt.metric]
        save_model(os.path.join(opt.save_dir, 'model_best.pth'), 
                   epoch, model)
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
  main(opt)