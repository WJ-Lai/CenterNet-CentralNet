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
    print('mergeuplayer', file=doc)
    print(model._modules['mergeuplayer'].alpha1.data, file=doc)
    print(model._modules['mergeuplayer'].alpha2.data, file=doc)
    print(model._modules['mergeuplayer'].alpha3.data, file=doc)
    print('', file=doc)
    print('mergeup_pre', file=doc)
    print(model._modules['mergeup_pre'].alpha1.data, file=doc)
    print(model._modules['mergeup_pre'].alpha2.data, file=doc)
    print(model._modules['mergeup_pre'].alpha3.data, file=doc)
    print('', file=doc)
    print('mergeup_kp_0', file=doc)
    print(model._modules['mergeup_kp']._modules['0'].alpha1.data, file=doc)
    print(model._modules['mergeup_kp']._modules['0'].alpha2.data, file=doc)
    print(model._modules['mergeup_kp']._modules['0'].alpha3.data, file=doc)
    print('', file=doc)
    print('mergeup_kp_1', file=doc)
    print(model._modules['mergeup_kp']._modules['1'].alpha1.data, file=doc)
    print(model._modules['mergeup_kp']._modules['1'].alpha2.data, file=doc)
    print(model._modules['mergeup_kp']._modules['1'].alpha3.data, file=doc)
    print('', file=doc)
    print('mergeup_head_0', file=doc)
    print(model._modules['mergeup_head']._modules['0'].alpha1.data, file=doc)
    print(model._modules['mergeup_head']._modules['0'].alpha2.data, file=doc)
    print(model._modules['mergeup_head']._modules['0'].alpha3.data, file=doc)
    print('mergeup_head_1', file=doc)
    print(model._modules['mergeup_head']._modules['1'].alpha1.data, file=doc)
    print(model._modules['mergeup_head']._modules['1'].alpha2.data, file=doc)
    print(model._modules['mergeup_head']._modules['1'].alpha3.data, file=doc)
    print('', file=doc)

class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)

def main(opt):
  # writer = SummaryWriter('../logs')

  torch.manual_seed(opt.seed)
  torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test

  # for fusion-----create new train loader
  Dataset2 = get_dataset('fir', opt.task)
  opt2 = opts().update_dataset_info_and_set_heads(opt, Dataset2)

  Dataset = get_dataset('rgb', opt.task)
  opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
  print(opt)

  logger = Logger(opt)

  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  opt.device = torch.device('cuda:'+opt.gpus_str if opt.gpus[0] >= 0 else 'cpu')
  os.environ['CUDA_VISIBLE_DEVICES'] = opt2.gpus_str
  opt2.device = opt.device

  print('Creating model...')
  model = create_model('hourglassfusion', opt.heads, opt.head_conv)
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
  val_loader2 = torch.utils.data.DataLoader(
      Dataset2(opt, 'val'),
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

  train_loader2 = torch.utils.data.DataLoader(
      Dataset2(opt2, 'train'),
      batch_size=opt.batch_size,
      shuffle=True,
      num_workers=opt.num_workers,
      pin_memory=True,
      drop_last=True
  )

  print('Starting training...')
  best = 1e10


  for epoch in range(start_epoch + 1, opt.num_epochs + 1):
    with open(opt.save_dir+'\\'+opt.exp_id+'-weight.txt', 'w+') as doc:
        print_weight(model, doc)
    mark = epoch if opt.save_all else 'last'
    log_dict_train, _ = trainer.train_fusion(epoch, train_loader, train_loader2)
    logger.write('epoch: {} |'.format(epoch))
    for k, v in log_dict_train.items():
      logger.scalar_summary('train_{}'.format(k), v, epoch)
      logger.write('{} {:8f} | '.format(k, v))
    if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
      save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(mark)), 
                 epoch, model, optimizer)
      with torch.no_grad():
        log_dict_val, preds = trainer.val(epoch, val_loader, val_loader2)
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