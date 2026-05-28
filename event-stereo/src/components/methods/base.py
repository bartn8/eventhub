import matplotlib.pyplot as plt
import cv2

from tqdm import tqdm
from collections import OrderedDict
import numpy as np

from utils.metric import AverageMeter, EndPointError, NPixelError, RootMeanSquareError, RelError, DeltaError
from utils import visualizer

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import time
import os

from thop import profile
import copy

_min_train_loss = -1
debug_img_counter = 0


def train(model, data_loader, optimizer, epoch=1, logger=None, log_interval=-1, scheduler=None, clip_grad_norm=False, args=None):
    global _min_train_loss
    # torch.autograd.set_detect_anomaly(True)
    model.train()

    log_dict = OrderedDict([
        ('Loss', AverageMeter(string_format='%8.5lf')),
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

    pbar = tqdm(data_loader, dynamic_ncols=True)
    for idx, batch_data in enumerate(pbar):
        batch_data = batch_to_cuda(batch_data)
        gt_disp = batch_data['disp']
        mask = gt_disp > 0

        # Get myconfidence, left, center, right RGB images
        myconfidence = batch_data['myconfidence'] if 'myconfidence' in batch_data.keys() else None
        rgb_left = batch_data['rgb_L'] if 'rgb_L' in batch_data.keys() else None
        rgb_center = batch_data['rgb_C'] if 'rgb_C' in batch_data.keys() else None
        rgb_right = batch_data['rgb_R'] if 'rgb_R' in batch_data.keys() else None

        H_gt, W_gt = gt_disp.shape[-2], gt_disp.shape[-1]
        
        # if not mask.any():
        #     continue

        event_stack_left = batch_data['event']['left']
        event_stack_right = batch_data['event']['right']

        # Rollback
        # B, T, C, H, W = event_stack_left.shape
        B, C, H, W = event_stack_left.shape

        assert H == H_gt and W == W_gt, f"Event shape {H}x{W} does not match GT shape {H_gt}x{W_gt}"

        _tensorboard_left = torch.sum(event_stack_left, 1) if event_stack_left.shape[1] != 3 else event_stack_left
        _tensorboard_left /= _tensorboard_left.max() if _tensorboard_left.max() > 0 else 1.0

        _tensorboard_right = torch.sum(event_stack_right, 1) if event_stack_right.shape[1] != 3 else event_stack_right
        _tensorboard_right /= _tensorboard_right.max() if _tensorboard_right.max() > 0 else 1.0

        pred, loss = model(left_stack=event_stack_left,
                        right_stack=event_stack_right,
                        gt_disparity=gt_disp,
                        confidence=myconfidence,
                        rgb_left=rgb_left,
                        rgb_right=rgb_right,
                        rgb_center=rgb_center,
                        test_mode=False,
                        iters=12)

        if isinstance(loss, float):
            # continue
            loss = torch.tensor(loss).float().to(pred.device)
            return log_dict
            

        optimizer.zero_grad()
        loss = loss.mean()

        loss.backward()
        if clip_grad_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        cur_lrs = [param_group['lr'] for param_group in optimizer.param_groups]
        pbar.set_description(f"Learning rate: {cur_lrs}; loss: {loss}")

        log_dict['Loss'].update(loss.item(), pred.size(0))

        # print(f"pred shape: {pred.shape}, gt_disp shape: {gt_disp.shape}, mask shape: {mask.shape}")
        gt_disp = gt_disp.squeeze(1) if gt_disp.dim() == 4 else gt_disp
        mask = mask.squeeze(1) if mask.dim() == 4 else mask

        if mask.any():
            for key in log_dict.keys():
                if key in ['EPE', '1PE', '2PE', '3PE', 'RMSE', 'REL', 'DEL1', 'DEL2', 'DEL3']:
                    # print(f"Updating metric {key}")
                    log_dict[key].update(pred, gt_disp, mask)

        # if log_interval != -1 and idx != 0 and idx % log_interval == 0 :
        if logger is not None and log_interval != -1 and idx % log_interval == 0:
            # # TODO: calculate here validation??
            # if _min_train_loss < 0 or _min_train_loss > loss.item():
            #     _min_train_loss = loss.item()
            #     if save_fn is not None:
            #         save_fn('best.pth')


            logger.add_scalar("train_steps/Loss", loss.item(), epoch*len(data_loader)+idx)
            
            if mask.any():
                for k in ['EPE', '1PE', '2PE', '3PE', 'RMSE', 'REL']:
                    logger.add_scalar("train_steps/%s"%k, log_dict[k].calculate_error(pred, gt_disp, mask), epoch*len(data_loader)+idx)
            
            _pred = pred[0].unsqueeze(0).detach()
            _pred = _pred if _pred.max() > _pred.min() else torch.zeros_like(pred[0].unsqueeze(0))
            logger.add_image("train_steps/pred", _pred, epoch*len(data_loader)+idx)
            logger.add_image("train_steps/gt", gt_disp[0].unsqueeze(0).detach(), epoch*len(data_loader)+idx)

            # for t in range(T):
            logger.add_image(f"train_steps/left_event", _tensorboard_left[0].unsqueeze(0).detach(), epoch*len(data_loader)+idx)
            logger.add_image(f"train_steps/right_event", _tensorboard_right[0].unsqueeze(0).detach(), epoch*len(data_loader)+idx)

        debug_img = True
        global debug_img_counter
        if args is not None and hasattr(args, 'save_root') and debug_img and debug_img_counter % 10 == 0:
            #Create a grid of images for debugging
            _top_rows = []
            # for t in range(T):            
            _a = _tensorboard_left[0].squeeze().cpu().numpy()
            _a = np.stack([_a,_a,_a], axis=-1) if _a.ndim == 2 else _a.transpose(1,2,0)
            _b = _tensorboard_right[0].squeeze().cpu().numpy()
            _b = np.stack([_b,_b,_b], axis=-1) if _b.ndim == 2 else _b.transpose(1,2,0)
            _top_row = (np.hstack([_a, _b])*255).astype(np.uint8)
            _top_rows.append(_top_row)
            _top_row = np.vstack(_top_rows)
            _c = pred[0].squeeze().detach().cpu().numpy()
            _c = (_c - _c.min()) / (_c.max() - _c.min()) if _c.max() > _c.min() else np.zeros_like(_c)
            _c = cv2.applyColorMap((_c*255).astype(np.uint8), cv2.COLORMAP_MAGMA)
            _d = gt_disp[0].squeeze().detach().cpu().numpy()
            _d = (_d - _d.min()) / (_d.max() - _d.min()) if _d.max() > _d.min() else np.zeros_like(_d)
            _d = cv2.applyColorMap((_d*255).astype(np.uint8), cv2.COLORMAP_MAGMA)
            _bottom_row_a = np.hstack([_c, np.zeros_like(_c)])
            _bottom_row_b = np.hstack([_d, np.zeros_like(_d)])
            _grid = np.vstack([_top_row, _bottom_row_a, _bottom_row_b])

            cv2.imwrite(os.path.join(args.save_root, "debug_img_train.png"), _grid)
        debug_img_counter += 1

    pbar.close()

    return log_dict

@torch.no_grad()
def test(model, data_loader, log_dict, args=None, seq_name=None, seq_log_dict = None):
    model.eval()
    pred_list = []
    epe_list = []
    
    flops_calculated = False
    runtime_queue = []
    mem_allocated_queue = []
    mem_reserved_queue = []
    mem_peak_queue = []

    
    with tqdm(data_loader, dynamic_ncols=True) as loader:
        idx=0
        for batch_data in loader:
            batch_data = batch_to_cuda(batch_data)

            mask = batch_data['disp'] > 0
            _save_predictions = args is not None and hasattr(args, 'save_predictions') and args.save_predictions
            if not mask.any() and not _save_predictions:
                continue

            event_stack_left = batch_data['event']['left']
            event_stack_right = batch_data['event']['right']

            # B, T, C, H, W = event_stack_left.shape
            B, C, H, W = event_stack_left.shape

            resize_height = args.resize_height if args is not None and hasattr(args, 'resize_height') and args.resize_height > 0 else None
            resize_width = args.resize_width if args is not None and hasattr(args, 'resize_width') and args.resize_width > 0 else None

            if resize_height is not None and resize_width is not None and resize_height > 0 and resize_width > 0:
                # for t in range(T):
                event_stack_left = F.interpolate(event_stack_left, (resize_height, resize_width), mode='nearest')
                event_stack_right = F.interpolate(event_stack_right, (resize_height, resize_width), mode='nearest')
                gt_disp = (F.interpolate(batch_data['disp'].unsqueeze(1), (resize_height, resize_width), mode='nearest') * (resize_width / W)).squeeze(1)
                mask = gt_disp > 0
            else:
                resize_width = W
                resize_height = H
                gt_disp = batch_data['disp']

            class ModelWrapper(nn.Module):
                def __init__(self, model):
                    super(ModelWrapper, self).__init__()
                    self.model = model

                def forward(self, left_stack, right_stack):
                    pred, _ = self.model(left_stack=left_stack,
                                        right_stack=right_stack,
                                        gt_disparity=None,
                                        test_mode=True,
                                        iters=22)
                    return pred

            if not flops_calculated:
                flops, params = profile(ModelWrapper(copy.deepcopy(model.module if isinstance(model, nn.DataParallel) else model),).to(event_stack_left.device),
                                        inputs=(event_stack_left, event_stack_right))
                print(f"FLOPs: {flops:.4f}, Params: {params:.4f} ")
                flops_calculated = True

            # # Reset delle statistiche della memoria GPU prima della forward
            # if torch.cuda.is_available():
            #     torch.cuda.reset_peak_memory_stats()
            #     torch.cuda.empty_cache()

            # start_time = time.time()

            pred, _ = model(left_stack=event_stack_left,
                            right_stack=event_stack_right,
                            gt_disparity=None,
                            test_mode=True,
                            iters=22)

            # end_time = time.time()
            # runtime_queue.append(end_time - start_time)
            # if len(runtime_queue) > 25:
            #     runtime_queue.pop(0)
            # avg_runtime = sum(runtime_queue) / len(runtime_queue)
            # print(f"Avg inference time: {avg_runtime*1000:.2f} ms")

            # # Stampa utilizzo VRAM
            # if torch.cuda.is_available():
            #     mem_allocated = torch.cuda.memory_allocated() / (1024 ** 2)
            #     mem_reserved = torch.cuda.memory_reserved() / (1024 ** 2)
            #     mem_peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
                
            #     mem_allocated_queue.append(mem_allocated)
            #     mem_reserved_queue.append(mem_reserved)
            #     mem_peak_queue.append(mem_peak)
            #     if len(mem_allocated_queue) > 25:
            #         mem_allocated_queue.pop(0)
            #         mem_reserved_queue.pop(0)
            #         mem_peak_queue.pop(0)
            #     avg_mem_allocated = sum(mem_allocated_queue) / len(mem_allocated_queue)
            #     avg_mem_reserved = sum(mem_reserved_queue) / len(mem_reserved_queue)
            #     avg_mem_peak = sum(mem_peak_queue) / len(mem_peak_queue)
            #     print(f"Avg VRAM allocata: {avg_mem_allocated:.2f} MB, riservata: {avg_mem_reserved:.2f} MB, picco: {avg_mem_peak:.2f} MB")
                
            #     # print(f"VRAM allocata: {mem_allocated:.2f} MB, riservata: {mem_reserved:.2f} MB, picco: {mem_peak:.2f} MB")
            
            gt_disp = gt_disp.squeeze(1) if gt_disp.dim() == 4 else gt_disp
            mask = mask.squeeze(1) if mask.dim() == 4 else mask  

            # Debug image creation for test
            debug_img = True
            global debug_img_counter
            if args is not None and hasattr(args, 'save_root') and debug_img and debug_img_counter % 10 == 0:
                #Create a grid of images for debugging
                _top_rows = []                
                # for t in range(T):
                _tensorboard_left = torch.sum(event_stack_left, 1) if event_stack_left.shape[1] != 3 else event_stack_left
                _tensorboard_left /= _tensorboard_left.max() if _tensorboard_left.max() > 0 else 1.0
                _tensorboard_right = torch.sum(event_stack_right, 1) if event_stack_right.shape[1] != 3 else event_stack_right
                _tensorboard_right /= _tensorboard_right.max() if _tensorboard_right.max() > 0 else 1.0

                _a = _tensorboard_left[0].squeeze().cpu().numpy()
                _a = np.stack([_a, _a, _a], axis=-1) if _a.ndim == 2 else _a.transpose(1, 2, 0)
                _b = _tensorboard_right[0].squeeze().cpu().numpy()
                _b = np.stack([_b, _b, _b], axis=-1) if _b.ndim == 2 else _b.transpose(1, 2, 0)
                _top_row = (np.hstack([_a, _b]) * 255).astype(np.uint8)
                _top_rows.append(_top_row)
                # Now _top_rows is a list of T _top_row images
                _top_row = np.vstack(_top_rows)
                _c = pred[0].squeeze().detach().cpu().numpy()
                _c = (_c - _c.min()) / (_c.max() - _c.min()) if _c.max() > _c.min() else np.zeros_like(_c)
                _c = cv2.applyColorMap((_c*255).astype(np.uint8), cv2.COLORMAP_MAGMA)
                _d = gt_disp[0].squeeze().detach().cpu().numpy()
                _d = (_d - _d.min()) / (_d.max() - _d.min()) if _d.max() > _d.min() else np.zeros_like(_d)
                _d = cv2.applyColorMap((_d*255).astype(np.uint8), cv2.COLORMAP_MAGMA)
                _bottom_row_a = np.hstack([_c, np.zeros_like(_c)])
                _bottom_row_b = np.hstack([_d, np.zeros_like(_d)])
                _grid = np.vstack([_top_row, _bottom_row_a, _bottom_row_b])

                cv2.imwrite(os.path.join(args.save_root, "debug_img_test.png"), _grid)
            debug_img_counter += 1

            if args is not None and hasattr(args, 'render') and args.render:
                width = data_loader.dataset.WIDTH
                height = data_loader.dataset.HEIGHT
                es_left = event_stack_left[0,:,:height,:width].cpu()
                es_right = event_stack_right[0,:,:height,:width].cpu()

                if es_left.shape[0] != 3:
                    es_left = torch.mean(es_left, dim=0)
                    es_right = torch.mean(es_right, dim=0)

                cur_pred = pred[0, :height, :width].cpu()
                cur_gt = gt_disp[0, :height, :width].cpu()

                #cur_errormap = visualizer.color_error_image_kitti(torch.abs(cur_gt-cur_pred).cpu().squeeze().numpy(), scale=0.5, mask=cur_gt>0, dilation=3)
                cur_errormap = visualizer.color_error_image_kitti(torch.abs(cur_gt-cur_pred).cpu().squeeze().numpy(), scale=1, mask=cur_gt>0, dilation=1)

                os.makedirs(os.path.join(args.save_root, "debug_images", seq_name), exist_ok=True)
                
                plt.imsave(os.path.join(args.save_root, "debug_images", seq_name, f"{idx:05d}_norm.jpg"), cur_pred, cmap='Spectral_r')
                plt.imsave(os.path.join(args.save_root, "debug_images", seq_name, f"{idx:05d}.jpg"), torch.clip(cur_pred, 0, cur_gt.max()), cmap='Spectral_r', vmin=0, vmax=cur_gt.max())
                _gt_img = cur_gt.squeeze().cpu().numpy()
                
                # Dilate ground truth by 7x7 kernel (max filter)
                # kernel = np.ones((5, 5), np.uint8)
                # _gt_img = _gt_img.astype(np.float32)
                # _gt_img = cv2.dilate(_gt_img, kernel)
                
                # create colored image with colormap but make zero values pure black
                if _gt_img.max() <= 0:
                    _img = np.zeros((_gt_img.shape[0], _gt_img.shape[1], 3), dtype=np.float32)
                else:
                    import matplotlib.cm as mcm
                    cmap = mcm.get_cmap('Spectral_r')
                    norm = _gt_img / _gt_img.max()
                    colored = cmap(norm)[:, :, :3]  # RGB float in [0,1]
                    colored[_gt_img == 0] = 0.0
                    _img = colored
                plt.imsave(os.path.join(args.save_root, "debug_images", seq_name, f"{idx:05d}_gt.png"), _img)
                cv2.imwrite(os.path.join(args.save_root, "debug_images", seq_name, f"{idx:05d}_errormap.png"), cur_errormap)

                if es_left.shape[0] != 3:
                    plt.imsave(os.path.join(args.save_root, "debug_images", seq_name, f"{idx:05d}_es_left.png"), es_left/es_left.max(), cmap='Spectral_r')
                    plt.imsave(os.path.join(args.save_root, "debug_images", seq_name, f"{idx:05d}_es_right.png"), es_right/es_right.max(), cmap='Spectral_r')
                else:
                    plt.imsave(os.path.join(args.save_root, "debug_images", seq_name, f"{idx:05d}_es_left.png"), es_left.permute(1,2,0).numpy())
                    plt.imsave(os.path.join(args.save_root, "debug_images", seq_name, f"{idx:05d}_es_right.png"), es_right.permute(1,2,0).numpy())

            idx+=1

            if resize_height is not None and resize_width is not None and resize_height > 0 and resize_width > 0:
                original_height, original_width = batch_data['disp'].shape[-2], batch_data['disp'].shape[-1]
                if original_height != resize_height or original_width != resize_width:
                    pred = (F.interpolate(pred.unsqueeze(0), (original_height, original_width), mode='bilinear') * (original_width / resize_width)).squeeze(0)
                    gt_disp = batch_data['disp']
                    mask = gt_disp > 0

            epe_list.append(log_dict['EPE'].calculate_error(pred, gt_disp, mask).item())

            if mask.any():
                for key in log_dict.keys():
                    if key in ['EPE', '1PE', '2PE', '3PE', 'RMSE', 'REL', 'DEL1', 'DEL2', 'DEL3']:
                        # print(f"Updating metric {key}")
                        log_dict[key].update(pred, gt_disp, mask)
                
                if seq_log_dict is not None:
                    for key in seq_log_dict.keys():
                        if key in ['EPE', '1PE', '2PE', '3PE', 'RMSE', 'REL', 'DEL1', 'DEL2', 'DEL3']:
                            # print(f"Updating metric {key}")
                            seq_log_dict[key].update(pred, gt_disp, mask)
                
            loader.set_description('Sequence %s, 1PE: %.4f'%(seq_name, log_dict['1PE'].calculate_error(pred, gt_disp, mask).item()))

            #Save prediction in KITTI format
            if args is not None and hasattr(args, 'save_predictions') and args.save_predictions:
                width = data_loader.dataset.WIDTH
                height = data_loader.dataset.HEIGHT
                cur_pred = pred[0, :height, :width].cpu().numpy()
                quantized_pred = np.clip(cur_pred*256.0, 0, 65535).astype(np.uint16)

                os.makedirs(os.path.join(args.save_root, "predictions", seq_name), exist_ok=True)
                cv2.imwrite(os.path.join(args.save_root, "predictions", seq_name, str(idx).zfill(6) + '.png'), quantized_pred)

    if args is not None and hasattr(args, 'save_root'):
        os.makedirs(os.path.join(args.save_root, "epes"), exist_ok=True)
        _file_path = os.path.join(args.save_root, "epes", f'{seq_name}epe_list.txt')
        if os.path.exists(_file_path):
            _tmp_data = np.loadtxt(_file_path)
            _tmp_data = _tmp_data.flatten().tolist() if _tmp_data.ndim > 0 else [_tmp_data.item()]
            epe_list = _tmp_data + epe_list
        np.savetxt(os.path.join(args.save_root, "epes", f'{seq_name}epe_list.txt'), np.array(epe_list))
            
    return pred_list


def batch_to_cuda(batch_data):
    def _batch_to_cuda(batch_data):
        if isinstance(batch_data, dict):
            for key in batch_data.keys():
                batch_data[key] = _batch_to_cuda(batch_data[key])
        elif isinstance(batch_data, torch.Tensor):
            batch_data = batch_data.cuda()
        else:
            raise NotImplementedError

        return batch_data

    for domain in ['event']:
        if domain not in batch_data.keys():
            batch_data[domain] = {}
        for location in ['left', 'right']:
            if location in batch_data[domain].keys():
                batch_data[domain][location] = _batch_to_cuda(batch_data[domain][location])
            else:
                batch_data[domain][location] = None

    for key in ['disp','rgb_C', 'rgb_L', 'rgb_R', 'ao', 'alpha', 'sizeconf', 'myconfidence']:
        if key in batch_data.keys() and batch_data[key] is not None:
            batch_data[key] = batch_data[key].cuda()

    return batch_data
