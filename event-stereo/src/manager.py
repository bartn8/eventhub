import os
import copy

import torch
import torch.nn as nn
import torch.optim as optim

from collections import OrderedDict
from utils.metric import AverageMeter, EndPointError, NPixelError, RootMeanSquareError, RelError, DeltaError

from components import models
from components.datasets import get_dataset, get_dataloader
from components import methods

from utils.logger import ExpLogger, TimeCheck
from utils.metric import SummationMeter, Metric, ListAverageMeter

from yacs.config import CfgNode as CN


class DLManager:
    def __init__(self, args, cfg=None, train_mode=True):
        self.args = args
        self.cfg = cfg
        self.train_mode = train_mode
        self.logger = ExpLogger(save_root=args.save_root)
        if self.cfg is not None:
            self.log_interval = self.cfg.LOG_INTERVAL

        if self.cfg is not None:
            self._init_from_cfg(cfg)

        self.current_epoch = 0
        self.best_val = None

    def _init_from_cfg(self, cfg:CN, crop_height = None, crop_width = None, num_events = None, test_batch_size = None, split_val = None, sampling_ratio_val = None):
        assert cfg is not None
        self.cfg = cfg

        self.cfg.defrost()# ------------------------------------------------------------------------------

        if crop_height is not None and crop_height > 0:
            self.cfg.DATASET.TEST.PARAMS.crop_height = crop_height

        if crop_width is not None and crop_width > 0:
            self.cfg.DATASET.TEST.PARAMS.crop_width = crop_width

        if num_events is not None and num_events > 0:
            self.cfg.DATASET.TEST.PARAMS.event_cfg.PARAMS.num_of_event = num_events

        if test_batch_size is not None:
            self.cfg.DATALOADER.TEST.PARAMS.batch_size = test_batch_size

        if split_val is not None:
            self.cfg.DATASET.TEST.PARAMS.split = split_val
            
        if sampling_ratio_val is not None:
            self.cfg.DATASET.TEST.PARAMS.sampling_ratio = sampling_ratio_val

        self.cfg.freeze()# --------------------------------------------------------------------------------

        self.model = _prepare_model(self.cfg.MODEL)

        if self.train_mode:
            self.optimizer = _prepare_optimizer(self.cfg.OPTIMIZER, self.model)
                        
            #check if self.cfg.DATASET.TRAIN is not a list
            dataset_cfg_list =  [self.cfg.DATASET.TRAIN] if not isinstance(self.cfg.DATASET.TRAIN, list) else self.cfg.DATASET.TRAIN
           
            self.datasets = []

            for dataset_cfg in dataset_cfg_list:
                #Convert dataset_cfg to a compatible format
                dataset_cfg = CN(dataset_cfg)
                
                _tmp_dataset = get_dataset(dataset_cfg.NAME, self.args, dataset_cfg)
                self.datasets.append(_tmp_dataset)

            # Concat datasets
            for dataset in self.datasets:
                print(f"Dataset with {len(dataset)} samples")

            dataset = torch.utils.data.ConcatDataset(self.datasets)
            self.data_loader = get_dataloader(self.cfg.DATALOADER.TRAIN.NAME, self.args, dataset, self.cfg.DATALOADER.TRAIN)

            _val_args = copy.deepcopy(self.args)
            _val_args.data_root = self.args.data_root_validation
            _val_args.num_workers = 1
            _val_args.seq_size = self.args.seq_size_val if self.args.seq_size_val >= 0 else self.args.seq_size
            
            dataset_val = get_dataset(self.cfg.DATASET.TEST.NAME, _val_args, self.cfg.DATASET.TEST)
            self.data_loader_validation = get_dataloader(self.cfg.DATALOADER.TEST.NAME, _val_args, dataset_val, self.cfg.DATALOADER.TEST)

            self.scheduler, self.do_epoch_step_scheduler = _prepare_scheduler(self.cfg.SCHEDULER, self.cfg.OPTIMIZER, self.optimizer, dataset_len=len(self.data_loader), epochs=self.cfg.TOTAL_EPOCH)
        else:    
            dataset = get_dataset(self.cfg.DATASET.TEST.NAME, self.args, self.cfg.DATASET.TEST)
            self.data_loader = get_dataloader(self.cfg.DATALOADER.TEST.NAME, self.args, dataset, self.cfg.DATALOADER.TEST)

        self.method = getattr(methods, self.cfg.METHOD)

    def train(self):
        self._log_before_train()

        train_loader = self.data_loader
        val_loader = self.data_loader_validation

        time_checker = TimeCheck(self.cfg.TOTAL_EPOCH)
        time_checker.start()

        for epoch in range(self.current_epoch, self.cfg.TOTAL_EPOCH):
            
            train_log_dict = self.method.train(model=self.model,
                                               data_loader=train_loader,
                                               optimizer=self.optimizer,
                                               epoch=epoch, logger=self.logger, log_interval=self.log_interval,
                                               scheduler=self.scheduler if not self.do_epoch_step_scheduler else None,
                                               clip_grad_norm=self.cfg.OPTIMIZER.clip_grad_norm if hasattr(self.cfg.OPTIMIZER, 'clip_grad_norm') else False,
                                               args=self.args)
            

            # Validation here
            if self.args.validate:

                val_log_dict = OrderedDict([
                    ('EPE', EndPointError(average_by='image', string_format='%8.5lf')),
                    ('1PE', NPixelError(n=1, average_by='image', string_format='%8.5lf')),
                    ('2PE', NPixelError(n=2, average_by='image', string_format='%8.5lf')),
                    ('3PE', NPixelError(n=3, average_by='image', string_format='%8.5lf')),
                    ('RMSE', RootMeanSquareError(average_by='image', string_format='%8.5lf')),
                    ('REL', RelError(average_by='image', string_format='%8.5lf')),
                    ('DEL1', DeltaError(n=1, average_by='image', string_format='%8.5lf')),
                    ('DEL2', DeltaError(n=2, average_by='image', string_format='%8.5lf')),
                    ('DEL3', DeltaError(n=3, average_by='image', string_format='%8.5lf')),
                ]) 

                for sequence_dataloader in val_loader:
                    sequence_name = sequence_dataloader.dataset.sequence_name
                    self.method.test(model=self.model, data_loader=sequence_dataloader, log_dict=val_log_dict, args=self.args, seq_name=sequence_name, seq_log_dict=None)
                    
                self._log_after_epoch(epoch + 1, None, val_log_dict, 'val_epochs', save_checkpoint=False)

                if self.best_val is None or self.best_val > val_log_dict['EPE'].value:
                    self.best_val = val_log_dict['EPE'].value
                    self.save('best.pth')

            if self.do_epoch_step_scheduler:
                self.scheduler.step()
                
            self.current_epoch += 1
            self._log_after_epoch(epoch + 1, time_checker, train_log_dict, 'train_epochs')

    def test(self):
        test_loader = self.data_loader

        self.logger.test()

        log_dict = OrderedDict([
            ('EPE', EndPointError(average_by='image', string_format='%8.5lf')),
            ('1PE', NPixelError(n=1, average_by='image', string_format='%8.5lf')),
            ('2PE', NPixelError(n=2, average_by='image', string_format='%8.5lf')),
            ('3PE', NPixelError(n=3, average_by='image', string_format='%8.5lf')),
            ('RMSE', RootMeanSquareError(average_by='image', string_format='%8.5lf')),
            ('REL', RelError(average_by='image', string_format='%8.5lf')),
            ('DEL1', DeltaError(n=1, average_by='image', string_format='%8.5lf')),
            ('DEL2', DeltaError(n=2, average_by='image', string_format='%8.5lf')),
            ('DEL3', DeltaError(n=3, average_by='image', string_format='%8.5lf')),
        ])

        for sequence_dataloader in test_loader:
            sequence_name = sequence_dataloader.dataset.sequence_name

            seq_log_dict = OrderedDict([
                ('EPE', EndPointError(average_by='image', string_format='%8.5lf')),
                ('1PE', NPixelError(n=1, average_by='image', string_format='%8.5lf')),
                ('2PE', NPixelError(n=2, average_by='image', string_format='%8.5lf')),
                ('3PE', NPixelError(n=3, average_by='image', string_format='%8.5lf')),
                ('RMSE', RootMeanSquareError(average_by='image', string_format='%8.5lf')),
                ('REL', RelError(average_by='image', string_format='%8.5lf')),
                ('DEL1', DeltaError(n=1, average_by='image', string_format='%8.5lf')),
                ('DEL2', DeltaError(n=2, average_by='image', string_format='%8.5lf')),
                ('DEL3', DeltaError(n=3, average_by='image', string_format='%8.5lf')),
            ])

            self.method.test(model=self.model, data_loader=sequence_dataloader, log_dict=log_dict, args=self.args, seq_name=sequence_name, seq_log_dict=seq_log_dict)
            
            # for cur_pred_dict in sequence_pred_list:
            #     file_name = cur_pred_dict.pop('file_name')
            #     for key in cur_pred_dict:
            #         self.logger.save_visualize(image=cur_pred_dict[key],
            #                                    visual_type=key,
            #                                    sequence_name=os.path.join('test', sequence_name),
            #                                    image_name=file_name)

            if self.args.verbose:
                print(f"Sequence {sequence_name} metrics")
                for k in seq_log_dict.keys():
                    print('%s: %.5f'%(k,seq_log_dict[k].value))

        if self.args.verbose:
            print("-"*40)

        for k in log_dict.keys():
            print('%s: %.5f'%(k,log_dict[k].value))

        csv_file = self.args.csv_file
        if csv_file is not None:
            args_dict = vars(self.args)
            write_header = not os.path.exists(csv_file)
            
            with open(csv_file, "a") as f:
                dict_keys = list(args_dict.keys())
                result_keys = list(log_dict.keys())
                if write_header:
                    for key in dict_keys:
                        f.write(f"{key},")
                    
                    for key in result_keys[:-1]:
                        f.write(f"{key},")

                    f.write(f"{result_keys[-1]}\n")

                for key in dict_keys:
                    f.write(f"{args_dict[key]},")
                
                for key in result_keys[:-1]:
                    f.write(f"{log_dict[key]},")

                f.write(f"{log_dict[result_keys[-1]]}\n")

            
    def save(self, name):
        checkpoint = self._make_checkpoint()
        self.logger.save_checkpoint(checkpoint, name)

    def resume(self, name):
        assert self.train_mode, "Cannot resume in test mode"

        checkpoint = self.logger.load_checkpoint(name)

        self.current_epoch = checkpoint['epoch']
        self.model.module.load_state_dict(checkpoint['model'])
        
        try:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        except ValueError as e:
            print(f"Warning: {e}. Optimizer state_dict may not match the model's parameters. This can happen if the model architecture has changed since the checkpoint was saved.")

        if self.scheduler is None:
            self.scheduler, self.do_epoch_step_scheduler = _prepare_scheduler(self.cfg.SCHEDULER, self.cfg.OPTIMIZER, self.optimizer, dataset_len=len(self.data_loader), epochs=self.cfg.TOTAL_EPOCH)

        self.scheduler.load_state_dict(checkpoint['scheduler'])

    def load_pretrain(self, name):
        checkpoint = self.logger.load_checkpoint(name)
        

        # Default way to load the model: this is the case for checkpoints saved with this framework
        if 'model' in checkpoint and 'stereo_matching_net' in list(checkpoint['model'].keys())[0]:
            self.model.module.load_state_dict(checkpoint['model'])
            return
        
        # Try to load the model saved with other frameworks.
        _model_key_list = ['state_dict', 'model']
        
        for key in _model_key_list:
            if key in checkpoint:
                checkpoint = checkpoint[key]
                break

        _checkpoint_keys = list(checkpoint.keys())
        backbone_name = self.cfg.MODEL.PARAMS.backbone
        
        if backbone_name in ['FoundationStereo', 'StereoAnywhere']:
            use_lora = getattr(self.cfg.MODEL.PARAMS.disparity_estimator.PARAMS.args, 'use_lora', False)
        else:
            use_lora = False

        if len(_checkpoint_keys) > 0:
            # Remove 'module.' prefix
            _new_checkpoint = {key.replace('module.', ''): v for key, v in checkpoint.items()}
            
            if use_lora:
                print(f"LoRA adaptation detected. Adjusting checkpoint keys for LoRA layers.")
                # Handle LoRA adaptation - map old checkpoint keys to new model structure
                _lora_checkpoint = {}
                for key, value in _new_checkpoint.items():
                    # Assume that LoRA layers are added to attn.qkv layers
                    # Map old qkv structure to new qkv.qkv structure for LoRA
                    if 'attn.qkv.weight' in key and 'attn.qkv.qkv.weight' not in key:
                        new_key = key.replace('attn.qkv.weight', 'attn.qkv.qkv.weight')
                        _lora_checkpoint[new_key] = value
                    elif 'attn.qkv.bias' in key and 'attn.qkv.qkv.bias' not in key:
                        new_key = key.replace('attn.qkv.bias', 'attn.qkv.qkv.bias')
                        _lora_checkpoint[new_key] = value
                    else:
                        _lora_checkpoint[key] = value
                
                self.model.module.stereo_matching_net.load_state_dict(_lora_checkpoint, strict=False)
            else:
                self.model.module.stereo_matching_net.load_state_dict(_new_checkpoint, strict=False)
        else:
            raise KeyError(f"Checkpoint is empty. It contains no keys.")

    def load_inference(self, inference_cfg:CN):
        assert inference_cfg is not None
        assert self.args is not None

        checkpoint_path = self.args.checkpoint_path
        crop_height = self.args.crop_height
        crop_width = self.args.crop_width
        num_events = self.args.num_events
        test_batch_size = self.args.test_batch_size
        split = self.args.split
        sampling_ratio = self.args.sampling_ratio

        checkpoint = self.logger.load_checkpoint(checkpoint_path)

        # Get model configuration from checkpoint
        checkpoint_cfg = checkpoint['cfg']
        
        # Use inference_cfg as base and selectively update from checkpoint
        # Start with inference config structure
        final_cfg = inference_cfg.clone()
        final_cfg.defrost()
        
        # Manually copy compatible sections from checkpoint
        if 'MODEL' in checkpoint_cfg:
            final_cfg.MODEL = checkpoint_cfg.MODEL
        if 'METHOD' in checkpoint_cfg and not ('METHOD' in final_cfg):
            final_cfg.METHOD = checkpoint_cfg.METHOD
            
        final_cfg.freeze()

        self._init_from_cfg(final_cfg, crop_height, crop_width, num_events, test_batch_size, split_val=split, sampling_ratio_val=sampling_ratio)
        self.model.module.load_state_dict(checkpoint['model'])

    def _make_checkpoint(self):
        checkpoint = {
            'epoch': self.current_epoch,
            'args': self.args,
            'cfg': self.cfg,
            'model': self.model.module.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
        }

        return checkpoint

    def _log_before_train(self):
        self.logger.train()
        self.logger.save_args(self.args)
        self.logger.save_cfg(self.cfg)
        self.logger.log_model(self.model)
        self.logger.log_optimizer(self.optimizer)
        self.logger.save_src(os.path.dirname(os.path.abspath(__file__)))

    def _log_after_epoch(self, epoch, time_checker, log_dict, part, save_checkpoint=True):
        # Calculate Time
        if time_checker is not None:
            time_checker.update(epoch)

            # Log Time
            self.logger.write('Epoch: %d | time per epoch: %s | eta: %s' %
                            (epoch, time_checker.time_per_epoch, time_checker.eta))
        else:
            self.logger.write('Epoch: %d | time per epoch: N/A | eta: N/A' % epoch)

        # Log Learning Process
        log = '%5s' % part
        for key in log_dict.keys():
            log += ' | %s: %s' % (key, str(log_dict[key]))
            if isinstance(log_dict[key], SummationMeter) or isinstance(log_dict[key], Metric) or isinstance(log_dict[key], ListAverageMeter):
                self.logger.add_scalar('%s/%s' % (part, key), log_dict[key].value, epoch)
            else:
                self.logger.add_scalar('%s/%s' % (part, key), log_dict[key], epoch)
        self.logger.write(log=log)

        # Make Checkpoint
        if save_checkpoint:
            checkpoint = self._make_checkpoint()

            # Save Checkpoint
            self.logger.save_checkpoint(checkpoint, 'final.pth')
            if epoch % self.args.save_term == 0:
                self.logger.save_checkpoint(checkpoint, '%d.pth' % epoch)


def _prepare_model(model_cfg):
    name = model_cfg.NAME
    parameters = model_cfg.PARAMS

    model = getattr(models, name)(**parameters)
    model = nn.DataParallel(model).cuda()

    return model


def _prepare_optimizer(optimizer_cfg, model):
    name = optimizer_cfg.NAME
    parameters = optimizer_cfg.PARAMS
    learning_rate = parameters.lr

    params_group = model.module.get_params_group(learning_rate)

    # print(params_group)
    # exit()

    optimizer = getattr(optim, name)(params_group, **parameters)

    return optimizer


def _prepare_scheduler(scheduler_cfg, optimizer_cfg, optimizer, dataset_len=None, epochs=None):
    name = scheduler_cfg.NAME
    parameters = scheduler_cfg.PARAMS
    do_epoch_step = False

    if dataset_len is not None and epochs is not None:
        total_steps = dataset_len * epochs
        if 'total_steps' in parameters:
            print(f"Warning: 'total_steps' in scheduler parameters, overwriting with {total_steps}")

        parameters['total_steps'] = total_steps

    parameters['max_lr'] = optimizer_cfg.PARAMS.lr

    if name == 'CosineAnnealingWarmupRestarts':
        from utils.scheduler import CosineAnnealingWarmupRestarts
        scheduler = CosineAnnealingWarmupRestarts(optimizer, **parameters)
        do_epoch_step = True
    else:
        scheduler = getattr(optim.lr_scheduler, name)(optimizer, **parameters)

    return scheduler, do_epoch_step
