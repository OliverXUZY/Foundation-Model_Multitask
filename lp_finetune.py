import argparse
import os
import random
import time
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
import datasets
import models
from models import encoders, classifiers
import utils
import utils.optimizers as optimizers


def main(config):
  SEED = config.get('seed') or 0
  utils.log("seed: {}".format(SEED))
  random.seed(SEED)
  np.random.seed(SEED)
  torch.manual_seed(SEED)
  torch.cuda.manual_seed(SEED)
  
  # torch.backends.cudnn.enabled = True
  # torch.backends.cudnn.benchmark = True
  # torch.backends.cudnn.deterministic = True

  ##### Dataset #####
  
  # V = config['train_set_args']['n_view'] = config['V']
  # SV = config['train_set_args']['n_meta_view'] = 1
  V = SV = 1

  # meta-train
  train_set = datasets.make(config['dataset'], **config['train_set_args'])
  utils.log('meta-train dataset: split-{} {} (x{}), {}'.format(config['train_set_args']['split'],
    train_set[0][0].shape, len(train_set), train_set.n_class))
  
  TE = train_set.n_episode
  TY = train_set.n_way
  TS = train_set.n_shot
  TQ = train_set.n_query

  # query-set labels
  y = torch.arange(TY)[:, None]
  y = y.repeat(TE, SV, TQ).flatten()      # [TE * SV * TY * TQ]
  y = y.cuda()

  train_loader = DataLoader(train_set, TE, num_workers=8, pin_memory=True)

  # meta-val
  eval_val = False
  if config.get('val_set_args'):
    eval_val = True
    val_set = datasets.make(config['dataset'], **config['val_set_args'])
    utils.log('meta-val dataset: {} (x{}), {}'.format(
      val_set[0][0].shape, len(val_set), val_set.n_class))

    E = val_set.n_episode
    Y = val_set.n_way
    S = val_set.n_shot
    Q = val_set.n_query

    # query-set labels
    val_y = torch.arange(Y)[:, None]
    val_y = val_y.repeat(E, Q).flatten()  # [E * Y * Q]
    val_y = val_y.cuda()

    val_loader = DataLoader(val_set, E, num_workers=8, pin_memory=True)
  
  ##### Model and Optimizer #####
  #################   encoder
  start_epoch_from = 0
  config['encoder_args'] = config.get('encoder_args') or dict()
  enc = encoders.make(config['encoder'], **config['encoder_args'])
  ckpt = {
    'encoder': config['encoder'],
    'encoder_args':  config['encoder_args'],
  }

  ################# classifier
  config['classifier_args'] = config.get('classifier_args') or dict()
  config['classifier_args']['in_dim'] = enc.get_out_dim()
  clf = classifiers.make(config['classifier'], **config['classifier_args'])
  
  model = models.Model(enc, clf)

  ################# linear probing
  if config['LP'] and config['LP_FT']:
    ### linear probing then finetuning
    pass
  elif config['LP']:
    ### if linear probing only
    ### LP from scratch, fix encoder
    config['lp_args'] = config.get('lp_args') or dict()
    config['lp_args']['in_dim'] = enc.get_out_dim()
    config['lp_args']['n_way'] = train_set.n_class
    lp = classifiers.make('logistic', **config['lp_args'])
      
  elif config['LP_FT']:
    if config.get('ckpt'):
      ### load LP, then finetuning the whole model
      assert os.path.exists(os.path.join(config['path'], config['ckpt']))
      ckpt = torch.load(os.path.join(config['path'], config['ckpt']))
      lp = classifiers.make(ckpt['lp'], **ckpt['lp_args'])
      if lp is not None:
        utils.log("load lp from {}".format(config.get('path')))
        lp.load_state_dict(ckpt['lp_state_dict'])
    else:
      ### random init LP, then finetuning the whole model
      config['lp_args'] = config.get('lp_args') or dict()
      config['lp_args']['in_dim'] = enc.get_out_dim()
      config['lp_args']['n_way'] = train_set.n_class
      lp = classifiers.make('logistic', **config['lp_args'])

  else:
    raise Exception("Not specify training mode!")
    
  model_lp = models.Model(enc, lp)

  ##### Optimizer #####
  if config['LP'] and config['LP_FT']:
    pass
  elif config['LP']:
    ### LP from scratch, fix encoder
    utils.log("load optimizer for LP only")
    lp_optimizer = optimizers.make(config['lp_optimizer'], model_lp.head.parameters(), 
                                **config['lp_optimizer_args'])
  elif config.get('LP_FT'):
    ### load LP,  then finetuning the whole model
    utils.log("load optimizer for LP_FT")
    optimizer = optimizers.make(config['optimizer'], model_lp.parameters(), 
                            **config['optimizer_args'])


  start_epoch = 1
  max_va = 0.

  if args.efficient:
    model.go_efficient()

  if config.get('_parallel'):
    model = nn.DataParallel(model)

  utils.log('num params: {}'.format(utils.count_params(model)))
  utils.log('M: {}, m: {}'.format(config['train_set_args']['n_batch'], 
                                  (config['train_set_args']['n_shot'] + config['train_set_args']['n_query'])*config['train_set_args']['n_way']
                                  ))
  timer_elapsed, timer_epoch = utils.Timer(), utils.Timer()

  
  ### if linear probing, ckpt name only contains lp names as classifiers
  ckpt_name = '{}_{}_{}_{}y{}s_{}m_{}M'.format(
    config['dataset'], ckpt['encoder'], 'lp_logistic',
    config['train_set_args']['n_way'], config['train_set_args']['n_shot'], 
    (config['train_set_args']['n_shot'] + config['train_set_args']['n_query']) * config['train_set_args']['n_way'],
    config['train_set_args']['n_batch']
  )

  if args.tag is not None:
    ckpt_name += '[' + args.tag + ']'

  if config.get('save_path'):
    ckpt_path = os.path.join(config['save_path'], ckpt_name)
  else:
    ckpt_path = os.path.join('./save/clip', ckpt_name)
  if config.get('rm_path'):
    utils.ensure_path(ckpt_path)
  utils.make_path(ckpt_path)
  utils.set_log_path(ckpt_path)
  
  utils.log("save to path: {}".format(ckpt_path))

  writer = SummaryWriter(os.path.join(ckpt_path, 'tensorboard'))
  yaml.dump(config, open(os.path.join(ckpt_path, 'config.yaml'), 'w'))

  ##### Training and evaluation #####
  
  xent_loss = nn.CrossEntropyLoss().cuda()

  aves_keys = ['tl', 'ta', 'vl', 'va']
  trlog = dict()
  for k in aves_keys:
    trlog[k] = []

  # sets warmup schedule
  if config['optimizer_args'].get('warmup'):
    try:
      warmup_epochs = config['optimizer_args']['warmup_epochs']
      warmup_from = config['optimizer_args']['warmup_from']
      warmup_to = config['optimizer_args'].get('warmup_to')
    except:
      raise ValueError('must specify `warmup_epochs` and `warmup_from`.')
    if warmup_to is None:
      warmup_to = utils.decay_lr(
        warmup_epochs, config['n_epochs'], **config['optimizer_args'])
    utils.log('warm-up learning rate for {} epochs from {} to {}'.format(
      str(warmup_epochs), warmup_from, warmup_to))
  else:
    warmup_epochs = -1
    warmup_from = warmup_to = None
  
  # print(type(config['lp_n_epochs']))
  # assert(False)
  if config['LP']:
    for epoch in range(start_epoch, config['lp_n_epochs'] + 1):
      timer_epoch.start()
      aves = {k: utils.AverageMeter() for k in aves_keys}

      np.random.seed(epoch + SEED)

      ########## if linear probing
      ### if linear probing while fix encoder
      eval_val = False # do not run eval on fs-centroid
      model_lp.train()
      model_lp.enc.eval()
      # adjust learning rate
      lr_lp = utils.decay_lr(epoch, config['lp_n_epochs'], **config['lp_optimizer_args'])
      for param_group in lp_optimizer.param_groups:
        param_group['lr'] = lr_lp
      for idx, (s, q, c) in enumerate(
        tqdm(train_loader, desc='train', leave=False)):
        # warm up learning rate
        if epoch <= warmup_epochs:
          lr = utils.warmup(warmup_from, warmup_to, 
                            epoch, warmup_epochs, idx, len(train_loader))
          for param_group in lp_optimizer.param_groups:
            param_group['lr'] = lr

        s = s.cuda(non_blocking=True)             # [TE, SV, TY * TS, V, C, H, W]
        q = q.cuda(non_blocking=True)             # [TE, SV, TY * TQ, C, H, W]
        s = s.view(TE, SV, TY, TS, *s.shape[-4:]) # [TE, SV, TY, TS, V, C, H, W]

        assert s.dim() == 8                             # [E, SV, Y, S, V, C, H, W]
        assert q.dim() == 6                             # [E, QV, Y * Q, C, H, W]
        assert s.size(0) == q.size(0)
        assert q.size(1) in [1, s.size(1)]
        s = s.transpose(0, 1)                           # [SV, E, Y, S, V, C, H, W]
        q = q.transpose(0, 1)                           # [QV, E, Y * Q, C, H, W]
        # print(s.shape)
        # print(q.shape)

        # SV_lp, E_lp, Y_lp, S_lp, V_lp = s.shape[:-3]
        # QV_lp, _, YQ_lp = q.shape[:-3]
        # YSV_lp, Q_lp = Y_lp * S_lp * V_lp, YQ_lp // Y_lp

        s = s.flatten(0, -4)                          # [SV * E * Y * S * V, C, H, W]
        q = q.flatten(0, -4)                          # [QV * E * Y * Q, C, H, W]
        x = torch.cat([s, q])                         # [150, C, H, W]
        # print(x.shape)
        logits = model_lp(x)
        # print(logits.shape)                 # [SV * E * Y * S * V + QV * E * Y * Q, y_out(train_set.n_class)] 
        
        # print(c.shape)
        # print(c.repeat_interleave(TQ).shape)
        y_lp = torch.cat([c.flatten(),c.repeat_interleave(TQ)])
        y_lp = y_lp.cuda()
        loss = xent_loss(logits, y_lp)
        acc = utils.accuracy(logits, y_lp)
        aves['tl'].update(loss.item())
        aves['ta'].update(acc[0])

        lp_optimizer.zero_grad()
        loss.backward()
        lp_optimizer.step()
      writer.add_scalar('lr', lp_optimizer.param_groups[0]['lr'], epoch)

      for k, avg in aves.items():
        aves[k] = avg.item()
        trlog[k].append(aves[k])

      t_epoch = utils.time_str(timer_epoch.end())
      t_elapsed = utils.time_str(timer_elapsed.end())
      t_estimate = utils.time_str(timer_elapsed.end() / 
        (epoch - start_epoch + 1) * (config['lp_n_epochs'] - start_epoch + 1))

      # formats output
      log_str = '[{}/{}] train {:.4f}(C)|{:.2f}'.format(
        str(epoch + start_epoch_from), str(config['lp_n_epochs'] + start_epoch_from), aves['tl'], aves['ta'])
      writer.add_scalars('loss', {'train': aves['tl']}, epoch + start_epoch_from)
      writer.add_scalars('acc', {'train': aves['ta']}, epoch + start_epoch_from)
      if eval_val:
        log_str += ', val {:.4f}(C)|{:.2f}'.format(aves['vl'], aves['va'])
        writer.add_scalars('loss', {'val': aves['vl']}, epoch + start_epoch_from)
        writer.add_scalars('acc', {'val': aves['va']}, epoch + start_epoch_from)

      log_str += ', {} {}/{}'.format(t_epoch, t_elapsed, t_estimate)
      utils.log(log_str)

      # saves model and meta-data
      if config.get('_parallel'):
        model_ = model.module
      else:
        model_ = model

      ckpt = {
        'file': __file__,
        'config': config,
        'epoch': epoch,
        'max_va': max(max_va, aves['va']),

        'encoder': ckpt['encoder'],
        'encoder_args': ckpt['encoder_args'],
        'encoder_state_dict': model_.enc.state_dict(),

        'lp': 'logistic',
        'lp_args': config.get('lp_args') or dict(),
        'lp_state_dict': model_lp.head.state_dict(),

        'lp_optimizer': config['lp_optimizer'],
        'lp_optimizer_args': config['lp_optimizer_args'],
        'lp_optimizer_state_dict': lp_optimizer.state_dict(),
      }
      torch.save(ckpt, os.path.join(ckpt_path, 'lp_epoch-last.pth'))
      torch.save(trlog, os.path.join(ckpt_path, 'lp_trlog.pth'))
      if aves['va'] > max_va:
        max_va = aves['va']
        torch.save(ckpt, os.path.join(ckpt_path, 'lp_max-va.pth'))
      if config.get('save_epoch') and epoch % config['save_epoch'] == 0:
        torch.save(ckpt, os.path.join(ckpt_path, 'lp_epoch-{}.pth'.format(epoch + start_epoch_from)))

  elif config.get('LP_FT'):
    for epoch in range(start_epoch, config['n_epochs'] + 1):
      timer_epoch.start()
      aves = {k: utils.AverageMeter() for k in aves_keys}

      np.random.seed(epoch + SEED)

      ########## if linear probing
      ### if linear probing, but finetune all
      eval_val = True 
      model_lp.train()
      ## change BatchNorm:
      if "RN" in ckpt['encoder']:
          enc.apply(utils.set_bn_eval)
      # adjust learning rate
      lr = utils.decay_lr(epoch, config['n_epochs'], **config['optimizer_args'])
      for param_group in optimizer.param_groups:
        param_group['lr'] = lr
      for idx, (s, q, c) in enumerate(
        tqdm(train_loader, desc='train', leave=False)):
        # warm up learning rate
        if epoch <= warmup_epochs:
          lr = utils.warmup(warmup_from, warmup_to, 
                            epoch, warmup_epochs, idx, len(train_loader))
          for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        s = s.cuda(non_blocking=True)             # [TE, SV, TY * TS, V, C, H, W]
        q = q.cuda(non_blocking=True)             # [TE, SV, TY * TQ, C, H, W]
        s = s.view(TE, SV, TY, TS, *s.shape[-4:]) # [TE, SV, TY, TS, V, C, H, W]

        assert s.dim() == 8                             # [E, SV, Y, S, V, C, H, W]
        assert q.dim() == 6                             # [E, QV, Y * Q, C, H, W]
        assert s.size(0) == q.size(0)
        assert q.size(1) in [1, s.size(1)]
        s = s.transpose(0, 1)                           # [SV, E, Y, S, V, C, H, W]
        q = q.transpose(0, 1)                           # [QV, E, Y * Q, C, H, W]
        # print(s.shape)
        # print(q.shape)

        s = s.flatten(0, -4)                          # [SV * E * Y * S * V, C, H, W]
        q = q.flatten(0, -4)                          # [QV * E * Y * Q, C, H, W]
        x = torch.cat([s, q])                         # [150, C, H, W]
        # print(x.shape)
        logits = model_lp(x)
        # print(logits.shape)                 # [SV * E * Y * S * V + QV * E * Y * Q, y_out(train_set.n_class)] 
        
        # print(c.shape)
        # print(c.repeat_interleave(TQ).shape)
        y_lp = torch.cat([c.flatten(),c.repeat_interleave(TQ)])
        y_lp = y_lp.cuda()
        loss = xent_loss(logits, y_lp)
        acc = utils.accuracy(logits, y_lp)
        aves['tl'].update(loss.item())
        aves['ta'].update(acc[0])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

      writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

      # meta-val
      if eval_val:
        model.eval()
        np.random.seed(SEED)
      
        with torch.no_grad():
          for (s, q, _) in tqdm(val_loader, desc='val', leave=False):
            s = s.cuda(non_blocking=True)         # [E, 1, Y * S, 1, C, H, W]
            q = q.cuda(non_blocking=True)         # [E, 1, Y * Q, C, H, W]
            s = s.view(E, 1, Y, S, *s.shape[-4:]) # [E, 1, Y, S, 1, C, H, W]
            
            logits, _ = model(s, q)               # [E, 1, Y * Q, Y]
            logits = logits.flatten(0, -2)        # [E * Y * Q, Y]
            loss = xent_loss(logits, val_y)
            acc = utils.accuracy(logits, val_y)
            aves['vl'].update(loss.item())
            aves['va'].update(acc[0])

      for k, avg in aves.items():
        aves[k] = avg.item()
        trlog[k].append(aves[k])

      t_epoch = utils.time_str(timer_epoch.end())
      t_elapsed = utils.time_str(timer_elapsed.end())
      t_estimate = utils.time_str(timer_elapsed.end() / 
        (epoch - start_epoch + 1) * (config['n_epochs'] - start_epoch + 1))

      # formats output
      log_str = '[{}/{}] train {:.4f}(C)|{:.2f}'.format(
        str(epoch + start_epoch_from), str(config['n_epochs'] + start_epoch_from), aves['tl'], aves['ta'])
      writer.add_scalars('loss', {'train': aves['tl']}, epoch + start_epoch_from)
      writer.add_scalars('acc', {'train': aves['ta']}, epoch + start_epoch_from)
      if eval_val:
        log_str += ', val {:.4f}(C)|{:.2f}'.format(aves['vl'], aves['va'])
        writer.add_scalars('loss', {'val': aves['vl']}, epoch + start_epoch_from)
        writer.add_scalars('acc', {'val': aves['va']}, epoch + start_epoch_from)

      log_str += ', {} {}/{}'.format(t_epoch, t_elapsed, t_estimate)
      utils.log(log_str)

      # saves model and meta-data
      if config.get('_parallel'):
        model_ = model.module
      else:
        model_ = model

      ckpt = {
        'file': __file__,
        'config': config,
        'epoch': epoch,
        'max_va': max(max_va, aves['va']),

        'encoder': ckpt['encoder'],
        'encoder_args': ckpt['encoder_args'],
        'encoder_state_dict': model_.enc.state_dict(),

        'classifier': config['classifier'],
        'classifier_args': config['classifier_args'],
        'classifier_state_dict': model_.head.state_dict(),

        'lp': 'logistic',
        'lp_args': config.get('lp_args') or dict(),
        'lp_state_dict': model_lp.head.state_dict(),

        'optimizer': config['optimizer'],
        'optimizer_args': config['optimizer_args'],
        'optimizer_state_dict': optimizer.state_dict(),
      }


      torch.save(ckpt, os.path.join(ckpt_path, 'epoch-last.pth'))
      torch.save(trlog, os.path.join(ckpt_path, 'trlog.pth'))
      if aves['va'] > max_va:
        max_va = aves['va']
        torch.save(ckpt, os.path.join(ckpt_path, 'max-va.pth'))
      if config.get('save_epoch') and epoch % config['save_epoch'] == 0:
        torch.save(ckpt, os.path.join(ckpt_path, 'epoch-{}.pth'.format(epoch + start_epoch_from)))

      writer.flush()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--config', 
                      help='configuration file')
  parser.add_argument('--tag', 
                      help='auxiliary information', 
                      type=str, default=None)
  parser.add_argument('--gpu', 
                      help='gpu device number', 
                      type=str, default='0')
  parser.add_argument('--efficient', 
                      help='if True, enables gradient checkpointing',
                      action='store_true')
  parser.add_argument('--n_batch_train',
                      help='modify batch train num batch',
                      type=int)
  parser.add_argument('--n_shot',
                      help='num shot',
                      type=int)
  parser.add_argument('--sample_per_task',
                      help='sample_per_task',
                      type=int)
  parser.add_argument('--path', 
                      help='the path to saved model', 
                      type=str)
  parser.add_argument('--stdFT', 
                      default=False,
                      help='whether we use standard finetune', 
                      action='store_true')
  
  args = parser.parse_args()
  config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
   
  utils.log('{}y{}s_{}m_{}M'.format(
    config['train_set_args']['n_way'], 
    config['train_set_args']['n_shot'], 
    (config['train_set_args']['n_shot'] + config['train_set_args']['n_query']) * config['train_set_args']['n_way'],
    config['train_set_args']['n_batch']
    ))
  if len(args.gpu.split(',')) > 1:
    config['_parallel'] = True
    config['_gpu'] = args.gpu
  # config['stdFT'] = args.stdFT
  # utils.set_gpu(args.gpu)
  main(config)