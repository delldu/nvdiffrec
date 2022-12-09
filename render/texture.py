# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.

import os
import numpy as np
import torch
import torch.nn as nn
import nvdiffrast.torch as dr

from . import util
import pdb

######################################################################################
# Smooth pooling / mip computation with linear gradient upscaling
######################################################################################

class texture2d_mip_function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, texture):
        return util.avg_pool_nhwc(texture, (2,2))

    @staticmethod
    def backward(ctx, dout):
        # gy, gx = torch.meshgrid(torch.linspace(0.0 + 0.25 / dout.shape[1], 1.0 - 0.25 / dout.shape[1], dout.shape[1]*2, device="cuda"), 
        #                         torch.linspace(0.0 + 0.25 / dout.shape[2], 1.0 - 0.25 / dout.shape[2], dout.shape[2]*2, device="cuda"),
        #                         indexing='ij')
        gy, gx = torch.meshgrid(torch.linspace(0.0 + 0.25 / dout.shape[1], 1.0 - 0.25 / dout.shape[1], dout.shape[1]*2, device="cuda"), 
                                torch.linspace(0.0 + 0.25 / dout.shape[2], 1.0 - 0.25 / dout.shape[2], dout.shape[2]*2, device="cuda"))

        uv = torch.stack((gx, gy), dim=-1)
        return dr.texture(dout * 0.25, uv[None, ...].contiguous(), filter_mode='linear', boundary_mode='clamp')

########################################################################################################
# Simple texture class. A texture can be either 
# - A 3D tensor (using auto mipmaps)
# - A list of 3D tensors (full custom mip hierarchy)
########################################################################################################

class Texture2D(nn.Module):
     # Initializes a texture from image data.
     # Input can be constant value (1D array) or texture (3D array) or mip hierarchy (list of 3d arrays)
    def __init__(self, init, min_max=None):
        super(Texture2D, self).__init__()
        # init = tensor([0.5000, 0.5000, 0.5000], device='cuda:0')
        # min_max = None

        if isinstance(init, np.ndarray):
            init = torch.tensor(init, dtype=torch.float32, device='cuda')
        elif isinstance(init, list) and len(init) == 1:
            init = init[0]

        if isinstance(init, list):
            self.data = list(nn.Parameter(mip.clone().detach(), requires_grad=True) for mip in init)
        elif len(init.shape) == 4:
            self.data = nn.Parameter(init.clone().detach(), requires_grad=True)
        elif len(init.shape) == 3:
            self.data = nn.Parameter(init[None, ...].clone().detach(), requires_grad=True)
        elif len(init.shape) == 1: # True
            self.data = nn.Parameter(init[None, None, None, :].clone().detach(), requires_grad=True) # Convert constant to 1x1 tensor
            # self.data.size() -- [1, 1, 1, 3]
        else:
            assert False, "Invalid texture object"

        self.min_max = min_max

    # Filtered (trilinear) sample texture at a given location
    def texture2d_sample(self, texc, texc_deriv, filter_mode='linear-mipmap-linear'):
        # self.data.size() -- [1, 2048, 2048, 3]
        # texc.size() -- [1, 512, 512, 2]
        # texc_deriv.size() -- [1, 512, 512, 4]
        # filter_mode = 'linear-mipmap-linear'

        if isinstance(self.data, list): # False
            out = dr.texture(self.data[0], texc, texc_deriv, mip=self.data[1:], filter_mode=filter_mode)
        else:
            # self.data.shape -- [1, 2048, 2048, 3]
            if self.data.shape[1] > 1 and self.data.shape[2] > 1: # False
                mips = [self.data]
                while mips[-1].shape[1] > 1 and mips[-1].shape[2] > 1:
                    mips += [texture2d_mip_function.apply(mips[-1])]
                out = dr.texture(mips[0], texc, texc_deriv, mip=mips[1:], filter_mode=filter_mode)
            else:
                out = dr.texture(self.data, texc, texc_deriv, filter_mode=filter_mode) # === !!!
        return out

    def getRes(self):
        return self.getMips()[0].shape[1:3] # [1, 2048, 2048, 3] --> [2048, 2048]

    def getChannels(self):
        return self.getMips()[0].shape[3] # [1, 2048, 2048, 3] --> 3

    def getMips(self):
        # self.data.size() -- [1, 1, 1, 3] or [1, 2048, 2048, 3]
        if isinstance(self.data, list): # False
            return self.data
        else:
            return [self.data]

    # In-place clamp with no derivative to make sure values are in valid range after training
    def clamp_(self):
        if self.min_max is not None:
            for mip in self.getMips():
                for i in range(mip.shape[-1]):
                    mip[..., i].clamp_(min=self.min_max[0][i], max=self.min_max[1][i])

    # In-place clamp with no derivative to make sure values are in valid range after training
    def normalize_(self):
        with torch.no_grad(): # !!! clamp with no derivative
            for mip in self.getMips():
                mip = util.safe_normalize(mip)


########################################################################################################
# Convert texture to and from SRGB
########################################################################################################

def srgb_to_rgb(texture):
    return Texture2D(list(util.srgb_to_rgb(mip) for mip in texture.getMips()))

def rgb_to_srgb(texture):
    return Texture2D(list(util.rgb_to_srgb(mip) for mip in texture.getMips()))

########################################################################################################
# Utility functions for loading / storing a texture
########################################################################################################

def _load_mip2D(fn, lambda_fn=None, channels=None):
    imgdata = torch.tensor(util.load_image(fn), dtype=torch.float32, device='cuda')
    if channels is not None:
        imgdata = imgdata[..., 0:channels]
    if lambda_fn is not None:
        imgdata = lambda_fn(imgdata)
    return imgdata.detach().clone()

def load_texture2D(fn, lambda_fn=None, channels=None):
    base, ext = os.path.splitext(fn)
    if os.path.exists(base + "_0" + ext):
        mips = []
        while os.path.exists(base + ("_%d" % len(mips)) + ext):
            mips += [_load_mip2D(base + ("_%d" % len(mips)) + ext, lambda_fn, channels)]
        return Texture2D(mips)
    else:
        return Texture2D(_load_mip2D(fn, lambda_fn, channels))

def _save_mip2D(fn, mip, mipidx, lambda_fn):
    if lambda_fn is not None:
        data = lambda_fn(mip).detach().cpu().numpy()
    else:
        data = mip.detach().cpu().numpy()

    if mipidx is None:
        util.save_image(fn, data)
    else:
        base, ext = os.path.splitext(fn)
        util.save_image(base + ("_%d" % mipidx) + ext, data)

def save_texture2D(fn, tex, lambda_fn=None):
    if isinstance(tex.data, list):
        for i, mip in enumerate(tex.data):
            _save_mip2D(fn, mip[0,...], i, lambda_fn)
    else:
        _save_mip2D(fn, tex.data[0,...], None, lambda_fn)
