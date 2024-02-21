import os
import argparse
import random

import yaml
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
torch.utils.tensorboard
import datasets
import models
from models import encoders, classifiers
import utils


def main(config):
  random.seed(0)
  np.random.seed(0)
  torch.manual_seed(0)
  torch.cuda.manual_seed(0)

  # torch.backends.cudnn.enabled = True
  # torch.backends.cudnn.benchmark = True
  # torch.backends.cudnn.deterministic = True

  ##### Dataset #####
  # V = config['test_set_args']['n_view'] = config['V']
  # config['test_set_args']['n_meta_view'] = 1


  test_set = datasets.make(config['dataset'], **config['test_set_args'])
  utils.log('test dataset: {} (x{}), {}'.format(
    test_set[0][0].shape, len(test_set), test_set.n_class), filename='test.txt')

  E = test_set.n_episode
  Y = test_set.n_way
  S = test_set.n_shot
  Q = test_set.n_query

  # query-set labels
  y = torch.arange(Y)[:, None]
  y = y.repeat(E, Q).flatten()
  y = y.cuda()                # [E * Y * Q]

  test_loader = DataLoader(test_set, E, num_workers=8, pin_memory=True)


  ##### Model #####
  if config.get('ckpt'):
    ckpt = torch.load(os.path.join(config['path'], config['ckpt']))
    ckpt['encoder'] = config['encoder'] ## adding for MAML
    ckpt['encoder_args'] = config.get('encoder_args') or dict() ## adding for MAML
    print("zhuoyan=====: ", ckpt['encoder'])

    ## add for testing train_head
    if ckpt.get('wrapper_state_dict'):
      ckpt['encoder_args'] = ckpt.get('encoder_args') or dict()
      enc = encoders.make(ckpt['encoder'], **ckpt['encoder_args'])

      ckpt['wrap_args'] = ckpt.get('wrap_args') or dict()
      ckpt['wrap'] = ckpt.get('wrap') or 'OneLayerNN'
      wrap = encoders.make(ckpt['wrap'], in_dim = enc.get_out_dim(), **ckpt['wrap_args'])

      wrapper = encoders.make('wrapper', enc = enc, wrap = wrap)
      wrapper.load_state_dict(ckpt['wrapper_state_dict'])
      enc = wrapper
    elif ckpt.get('wrap_state_dict'):
      ckpt['encoder_args'] = ckpt.get('encoder_args') or dict()
      enc = encoders.make(ckpt['encoder'], **ckpt['encoder_args'])

      ckpt['wrap_args'] = ckpt.get('wrap_args') or dict()
      ckpt['wrap'] = ckpt.get('wrap') or 'OneLayerNN'
      wrap = encoders.make(ckpt['wrap'], in_dim = enc.get_out_dim(), **ckpt['wrap_args'])
      
      wrap.load_state_dict(ckpt['wrap_state_dict'])
      wrapper = encoders.make('wrapper', enc = enc, wrap = wrap)
      
      enc = wrapper
    else:
      # print("zhuoyan: ckpt['encoder_state_dict']: ", ckpt['encoder_state_dict'])
      enc = encoders.load(ckpt)
  else:
    config['encoder_args'] = config.get('encoder_args') or dict()
    enc = encoders.make(config['encoder'], **config['encoder_args'])
    ckpt = {
      'encoder': config['encoder'],
      'encoder_args':  config['encoder_args'],
      }
    
    if 'dinov2' in config['encoder']:
      modeldir = 'dinov2'
    elif 'clip' in config['encoder']:
      modeldir = 'clip'
    elif 'torchvision' in config['encoder']:
      modeldir = 'torchvision'
    elif 'mocov3' in config['encoder']:
      modeldir = 'mocov3'
    elif 'dino_vit' in config['encoder']:
      modeldir = 'dino'
    else:
      print("model dir not found for encoder {}!".format(config['encoder']))
    
    config['path'] = "./save/{}/{}/{}".format(modeldir, config['dataset'].replace('meta-', ''), "original")
    utils.log("construct encoder {} from pre-train".format(config['encoder']))

  # enc = encoders.load(ckpt)
  clf = classifiers.make(
    config['classifier'], **{'in_dim': enc.get_out_dim(), 'n_way': Y})
  model = models.Model(enc, clf)

  if config.get('_parallel'):
    model.enc = nn.DataParallel(model.enc)

  utils.make_path(config['path'])
  utils.set_log_path(config['path'])

  timer_elapsed, timer_epoch = utils.Timer(), utils.Timer()

  ##### Evaluation #####
  utils.log('{}_{}_{}_{}y{}s:'.format(
    config['dataset'],
    config['encoder'],
    config['classifier'], 
    config['test_set_args']['n_way'], config['test_set_args']['n_shot']), filename='test.txt')

  model.eval()
  aves_keys = ['va']
  aves = {k: utils.AverageMeter() for k in aves_keys}
  va_lst = []

  for epoch in range(1, config['n_epochs'] + 1):
    np.random.seed(epoch)

    with torch.no_grad():
      for (s, q, _) in tqdm(test_loader, desc='test', leave=False):
        s = s.cuda(non_blocking=True)
        q = q.cuda(non_blocking=True)
        s = s.view(E, 1, Y, S, *s.shape[-4:])
        
        logits, _ = model(s, q)
        logits = logits.flatten(0, -2)                  # [E * Y * Q, Y]
        acc = utils.accuracy(logits, y)
        aves['va'].update(acc[0])
        va_lst.append(acc[0].item())

    log_str = '[{}/{}]: acc={:.2f} +- {:.2f} (%)'.format(
      epoch, str(config['n_epochs']), aves['va'].item(), 
      utils.mean_confidence_interval(va_lst))
    
    t_epoch = utils.time_str(timer_epoch.end())
    t_elapsed = utils.time_str(timer_elapsed.end())
    t_estimate = utils.time_str(timer_elapsed.end() / 
      (epoch - 1 + 1) * (config['n_epochs'] - 1 + 1))

    log_str += ', {} {}/{}'.format(t_epoch, t_elapsed, t_estimate)
    utils.log(log_str, filename='test.txt')


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--config', 
                      help='configuration file')
  parser.add_argument('--gpu', 
                      help='gpu device number', 
                      type=str, default='0')
  parser.add_argument('--path', 
                      help='the path to saved model', 
                      type=str)
  parser.add_argument('--save_path', 
                      help='the path from home to saved model', 
                      type=str)
  parser.add_argument('--exp', 
                      help='type of experiments', 
                      type=str, default='Mm_trend')
  parser.add_argument('--n_shot',
                      help='num shot',
                      type=int)
  parser.add_argument('--n_way',
                      help='num of classes',
                      type=int)
  args = parser.parse_args()
  config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)

  if 'dinov2' in config['encoder']:
    modeldir = 'dinov2'
  elif 'clip' in config['encoder']:
    modeldir = 'clip'
  elif 'torchvision' in config['encoder']:
    modeldir = 'torchvision'
  elif 'mocov3' in config['encoder']:
    modeldir = 'mocov3'

  if args.path:  # customized saved path here
    config['path'] = "./save/{}/{}/{}/{}".format(
      modeldir,
      config['dataset'].replace('meta-', ''), 
      args.exp, 
      args.path
      )
    utils.log("load model from path: {}".format(config['path']))
  # print("zhuoyan: ", config['path'])
  if args.save_path:
    config['path'] = args.save_path
    utils.log("load model from path: {}".format(config['path']))
  
  if args.n_shot:
    config['test_set_args']['n_shot'] = int(args.n_shot)
  if args.n_way:
    config['test_set_args']['n_way'] = int(args.n_way)

  if len(args.gpu.split(',')) > 1:
    config['_parallel'] = True
    config['_gpu'] = args.gpu

  # utils.set_gpu(args.gpu)
  main(config)
