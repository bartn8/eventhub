import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import torch
import torch.nn as nn
import numpy as np
import cv2
import torch.nn.functional as F
import torchvision

def SSIM(x, y, md=3):
    patch_size = 2 * md + 1
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    refl = nn.ReflectionPad2d(md)

    x = refl(x)
    y = refl(y)
    mu_x = nn.AvgPool2d(patch_size, 1, 0)(x)
    mu_y = nn.AvgPool2d(patch_size, 1, 0)(y)
    mu_x_mu_y = mu_x * mu_y
    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)

    sigma_x = nn.AvgPool2d(patch_size, 1, 0)(x * x) - mu_x_sq
    sigma_y = nn.AvgPool2d(patch_size, 1, 0)(y * y) - mu_y_sq
    sigma_xy = nn.AvgPool2d(patch_size, 1, 0)(x * y) - mu_x_mu_y

    SSIM_n = (2 * mu_x_mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x_sq + mu_y_sq + C1) * (sigma_x + sigma_y + C2)
    SSIM = SSIM_n / SSIM_d
    dist = torch.clamp((1 - SSIM) / 2, 0, 1)
    return dist

def norm_grid(v_grid):
    _, _, H, W = v_grid.size()

    # scale grid to [-1,1]
    v_grid_norm = torch.zeros_like(v_grid)
    v_grid_norm[:, 0, :, :] = 2.0 * v_grid[:, 0, :, :] / (W - 1) - 1.0
    v_grid_norm[:, 1, :, :] = 2.0 * v_grid[:, 1, :, :] / (H - 1) - 1.0
    return v_grid_norm.permute(0, 2, 3, 1) 

def mesh_grid(B, H, W):
    # mesh grid
    x_base = torch.arange(0, W).repeat(B, H, 1)
    y_base = torch.arange(0, H).repeat(B, W, 1).transpose(1, 2) 

    base_grid = torch.stack([x_base, y_base], 1)
    return base_grid

def gradient(data):
    D_dy = data[:, :, 1:] - data[:, :, :-1]
    D_dx = data[:, :, :, 1:] - data[:, :, :, :-1]
    return D_dx, D_dy

def smooth_grad(disp, image, alpha, order=1):
    img_dx, img_dy = gradient(image)
    weights_x = torch.exp(-torch.mean(torch.abs(img_dx), 1, keepdim=True) * alpha)
    weights_y = torch.exp(-torch.mean(torch.abs(img_dy), 1, keepdim=True) * alpha)

    dx, dy = gradient(disp)
    if order == 2:
        dx2, dxdy = gradient(dx)
        dydx, dy2 = gradient(dy)
        dx, dy = dx2, dy2

    loss_x = weights_x[:, :, :, 1:] * dx[:, :, :, 1:].abs()
    loss_y = weights_y[:, :, 1:, :] * dy[:, :, 1:, :].abs()

    return loss_x.mean() / 2. + loss_y.mean() / 2.

def loss_smooth(disp, im1_scaled):
    func_smooth = smooth_grad
    loss = []
    loss += [func_smooth(disp, im1_scaled, 1, order=1)]
    return sum([l.mean() for l in loss])

def disp_warp(x, disp, r2l=False, pad='border', mode='bilinear', device='cuda'):
    B, _, H, W = x.size()
    offset = -1
    if r2l:
        offset = 1

    base_grid = mesh_grid(B, H, W).type_as(x)
    v_grid = norm_grid(base_grid + torch.cat((offset*disp,torch.zeros_like(disp)),1)) 
    x_recons = nn.functional.grid_sample(x, v_grid, mode=mode, padding_mode=pad, align_corners=True)
    mask = torch.autograd.Variable(torch.ones(x_recons.size())).to(device)
    mask = nn.functional.grid_sample(mask, v_grid, align_corners=True)
    return x_recons, mask

def photometric_loss(im1_scaled, im1_recons, weight_l1 = 0.15, weight_ssim = 0.85):
    loss = []
    loss += [weight_l1 * (im1_scaled - im1_recons).abs().mean(1, True)]
    loss += [weight_ssim * SSIM(im1_recons, im1_scaled).mean(1, True)]
    return sum([l for l in loss])


def trinocular_loss(disp, im1, im2, im3, uncertainty):
    assert im1.shape == im2.shape == im3.shape, "Input images must have the same shape"
    assert uncertainty.dim() in [3, 4], "Uncertainty must be a 3D or 4D tensor"
    uncertainty = uncertainty.squeeze(1) if uncertainty.dim() == 4 else uncertainty

    im2_recons_from_1, mask_12 = disp_warp(im1, disp, r2l=True)
    im2_recons_from_3, mask_23 = disp_warp(im3, disp, r2l=False)

    photometric_loss_12 = photometric_loss(im2, mask_12 * im2_recons_from_1)
    photometric_loss_23 = photometric_loss(im2, mask_23 * im2_recons_from_3)
    loss_warp, _ = torch.min(torch.cat((photometric_loss_12, photometric_loss_23), dim=1), dim=1)

    photometric_loss_1 = photometric_loss(im2, im1)
    photometric_loss_3 = photometric_loss(im2, im3)
    loss_2, _ = torch.min(torch.cat((photometric_loss_1, photometric_loss_3), dim=1), dim=1)

    automask = loss_warp < loss_2

    #print(f"loss_warp shape: {loss_warp.shape}, loss_2 shape: {loss_2.shape}, uncertainty shape: {uncertainty.shape}, automask shape: {automask.shape}")

    loss = (loss_warp * uncertainty)
    loss[~automask] = 0

    return loss

def binocular_loss(disp, im1, im2, uncertainty):
    im1_recons, _ = disp_warp(im2, disp, r2l=False)

    loss_warp = photometric_loss(im1, im1_recons)
    loss_2 = photometric_loss(im2, im1)

    automask = loss_warp < loss_2
    loss = (loss_warp * uncertainty)
    loss[~automask] = 0

    return loss

def image_loss(disp, im1, im2, im3, uncertainty, trinocular=True):
    if trinocular:
        return trinocular_loss(disp, im1, im2, im3, uncertainty)
    else:
        return binocular_loss(disp, im2, im3, uncertainty)



## For training

# target_disp = data["label"]
# conf = data["conf"] * (target_disp > 0).float()

# n_predictions = len(pred_disps)
# loss_gamma = 0.9
# target_disp = target_disp.unsqueeze(1)

# disp_loss = 0.0
# photometric_loss = 0.0

# #  PSMNet & CFNet
# for i in range(len(pred_disps)):
#     disp_loss += (1 / 2 ** i) * (torch.abs(pred_disps[i] - target_disp) * conf * (target_disp > 0).float()).mean()
#     photometric_loss += (1 / 2 ** i) * image_loss(pred_disps[i].unsqueeze(1), data['im0'], data['im1'], data['im2'], 1 - conf, args.trinocular_loss)

# loss = args.alpha_disp_loss * disp_loss + args.alpha_photometric * photometric_loss

# # RAFT-Stereo
# for i in range(n_predictions):
#     adjusted_loss_gamma = loss_gamma ** (15 / (n_predictions - 1))
#     i_weight = adjusted_loss_gamma ** (n_predictions - i - 1)
    
#     disp_diff = torch.abs(pred_disps[i] - target_disp)
#     disp_loss += i_weight * (disp_diff * conf * (target_disp > 0).float()).mean()
    
#     if args.alpha_photometric != 0.:
#         photometric_loss += i_weight * image_loss(pred_disps[i], data['im0'], data['im1'], data['im2'], 1 - conf, args.trinocular_loss)
#     else:
#         photometric_loss = torch.zeros_like(disp_loss)

# loss = args.alpha_disp_loss * disp_loss + args.alpha_photometric * photometric_loss



