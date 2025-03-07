import torch
import torch.nn as nn
from torch import autograd
import math
import time
import numpy as np
import os
np.bool = np.bool_


def calculate_psnr(img1, img2):
    # img1: img
    # img2: gt
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    mse = np.mean((img1 - img2)**2)
    psnr = 20 * math.log10(255.0 / math.sqrt(mse))

    return psnr

    
def sg_ve_estimation_loss(model,
                          x_img: torch.Tensor,
                          x_gt: torch.Tensor,
                          sigma: torch.Tensor,
                          embedding: torch.Tensor,
                          e: torch.Tensor,
                          keepdim=False):

    # X_T
    sigma = sigma.to(torch.float32)
    embedding = embedding.to(torch.float32)
    x = x_gt + e * sigma.view(-1, 1, 1, 1)
    output = model(torch.cat([x_img, x], dim=1), embedding)

    if keepdim:
        return (e - output).square().sum(dim=(1, 2, 3))
    else:
        return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)

     
def sr_ve_estimation_loss(model,
                          x_bw: torch.Tensor,
                          x_md: torch.Tensor,
                          x_fw: torch.Tensor,
                          sigma: torch.Tensor,
                          embedding: torch.Tensor,
                          e: torch.Tensor,
                          keepdim=False):

    # X_T
    sigma = sigma.to(torch.float32)
    embedding = embedding.to(torch.float32)
    x = x_md + e * sigma.view(-1, 1, 1, 1)
    output = model(torch.cat([x_bw, x_fw, x], dim=1), embedding)
    
    if keepdim:
        return (e - output).square().sum(dim=(1, 2, 3))
    else:
        return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)    
    
    
loss_registry = {
    'sg_ve': sg_ve_estimation_loss,
    'sr_ve': sr_ve_estimation_loss
}




