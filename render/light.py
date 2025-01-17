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
from . import renderutils as ru
import pdb

######################################################################################
# Utility functions
######################################################################################

class cubemap_mip_function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, cubemap):
        return util.avg_pool_nhwc(cubemap, (2,2))

    @staticmethod
    def backward(ctx, dout):
        res = dout.shape[1] * 2
        out = torch.zeros(6, res, res, dout.shape[-1], dtype=torch.float32, device="cuda")
        for s in range(6):
            # gy, gx = torch.meshgrid(torch.linspace(-1.0 + 1.0 / res, 1.0 - 1.0 / res, res, device="cuda"), 
            #                         torch.linspace(-1.0 + 1.0 / res, 1.0 - 1.0 / res, res, device="cuda"),
            #                         indexing='ij')
            gy, gx = torch.meshgrid(torch.linspace(-1.0 + 1.0 / res, 1.0 - 1.0 / res, res, device="cuda"), 
                                    torch.linspace(-1.0 + 1.0 / res, 1.0 - 1.0 / res, res, device="cuda"))

            v = util.safe_normalize(util.cube_to_dir(s, gx, gy))

            # Perform texture sampling on dout
            out[s, ...] = dr.texture(dout[None, ...] * 0.25, v[None, ...].contiguous(), filter_mode='linear', boundary_mode='cube')
        return out

######################################################################################
# Split-sum environment map light source with automatic mipmap generation
######################################################################################

class EnvironmentLight(nn.Module):
    LIGHT_MIN_RES = 16

    MIN_ROUGHNESS = 0.08
    MAX_ROUGHNESS = 0.5

    def __init__(self, base):
        super(EnvironmentLight, self).__init__()
        self.base = nn.Parameter(base.clone().detach(), requires_grad=True)
        # self.register_parameter('env_base', self.base) # xxxx8888 !!!
        # self.base.size() -- [6, 512, 512, 3]

    def clone(self):
        return EnvironmentLight(self.base.clone().detach())

    def clamp_(self, min=None, max=None):
        self.base.clamp_(min, max)

    def get_mip(self, roughness):
        # roughness.size() -- [1, 512, 512, 1]
        x = torch.where(roughness < self.MAX_ROUGHNESS
                        , (torch.clamp(roughness, self.MIN_ROUGHNESS, self.MAX_ROUGHNESS) - self.MIN_ROUGHNESS) / (self.MAX_ROUGHNESS - self.MIN_ROUGHNESS) * (len(self.specular) - 2)
                        , (torch.clamp(roughness, self.MAX_ROUGHNESS, 1.0) - self.MAX_ROUGHNESS) / (1.0 - self.MAX_ROUGHNESS) + len(self.specular) - 2)
        return x # [1, 512, 512, 1], data == 1.6190
        
    def build_mips(self, cutoff=0.99):
        self.specular = [self.base] # self.base.size() -- [6, 512, 512, 3]
        while self.specular[-1].shape[1] > self.LIGHT_MIN_RES:
            self.specular += [cubemap_mip_function.apply(self.specular[-1])]

        self.diffuse = ru.cubemap_diffuse(self.specular[-1]) # self.specular[-1].size() -- [6, 16, 16, 3]

        for idx in range(len(self.specular) - 1):
            roughness = (idx / (len(self.specular) - 2)) * (self.MAX_ROUGHNESS - self.MIN_ROUGHNESS) + self.MIN_ROUGHNESS
            self.specular[idx] = ru.cubemap_specular(self.specular[idx], roughness, cutoff) 
        self.specular[-1] = ru.cubemap_specular(self.specular[-1], 1.0, cutoff)

        # len(self.specular) -- 6
        # (Pdb) for i in range(6): print(i, ":", self.specular[i].shape)
        # 0 : torch.Size([6, 512, 512, 3])
        # 1 : torch.Size([6, 256, 256, 3])
        # 2 : torch.Size([6, 128, 128, 3])
        # 3 : torch.Size([6, 64, 64, 3])
        # 4 : torch.Size([6, 32, 32, 3])
        # 5 : torch.Size([6, 16, 16, 3])        

    def regularizer(self):
        white = (self.base[..., 0:1] + self.base[..., 1:2] + self.base[..., 2:3]) / 3.0
        return torch.mean(torch.abs(self.base - white))

    def shade(self, gb_pos, gb_normal, kd, ks, view_pos, specular=True):
        # gb_pos.size() -- [1, 512, 512, 3], range [-1.0, 1.0]
        # gb_normal.size() -- [1, 512, 512, 3], range [-1.0, 1.0]
        # (Pdb) kd.size() -- [1, 512, 512, 3]
        # (Pdb) ks.size() -- [1, 512, 512, 3]
        # view_pos.size() -- [1, 1, 1, 3], data == [[[[ 1.2133, -1.9934, -1.6240]]]]

        wo = util.safe_normalize(view_pos - gb_pos)

        if specular: # True
            roughness = ks[..., 1:2] # y component
            metallic  = ks[..., 2:3] # z component
            spec_col  = (1.0 - metallic)*0.04 + kd * metallic # -- ks in paper ?
            diff_col  = kd * (1.0 - metallic)
        else:
            diff_col = kd

        reflvec = util.safe_normalize(util.reflect(wo, gb_normal))

        # Diffuse lookup
        # Perform texture sampling on self.diffuse
        diffuse = dr.texture(self.diffuse[None, ...], gb_normal.contiguous(), filter_mode='linear', boundary_mode='cube')
        shaded_color = diffuse * diff_col

        if specular: # True
            # Lookup FG term from lookup texture
            NdotV = torch.clamp(util.dot(wo, gb_normal), min=1e-4)
            fg_uv = torch.cat((NdotV, roughness), dim=-1)
            if not hasattr(self, '_FG_LUT'): # True
                self._FG_LUT = torch.as_tensor(np.fromfile('data/irrmaps/bsdf_256_256.bin', dtype=np.float32).reshape(1, 256, 256, 2), dtype=torch.float32, device='cuda')
            # self._FG_LUT.size() -- [1, 256, 256, 2]
            # Perform texture sampling on self._FG_LUT
            fg_lookup = dr.texture(self._FG_LUT, fg_uv, filter_mode='linear', boundary_mode='clamp')

            # Roughness adjusted specular env lookup
            miplevel = self.get_mip(roughness)
            spec = dr.texture(self.specular[0][None, ...], reflvec.contiguous(), mip=list(m[None, ...] for m in self.specular[1:]), mip_level_bias=miplevel[..., 0], filter_mode='linear-mipmap-linear', boundary_mode='cube')

            # Compute aggregate lighting
            reflectance = spec_col * fg_lookup[...,0:1] + fg_lookup[...,1:2]
            shaded_color += spec * reflectance

        return shaded_color * (1.0 - ks[..., 0:1]) # Modulate by hemisphere visibility

######################################################################################
# Load and store
######################################################################################

# Load from latlong .HDR file
def _load_env_hdr(fn, scale=1.0):
    # fn = 'data/irrmaps/aerodynamics_workshop_2k.hdr'
    # scale = 2.0
    latlong_img = torch.tensor(util.load_image(fn), dtype=torch.float32, device='cuda')*scale
    # (Pdb) latlong_img.size() -- [1024, 2048, 3], latlong_img.min() -- 0., latlong_img.max() -- 110.5000 !!!
    cubemap = util.latlong_to_cubemap(latlong_img, [512, 512])
    # (Pdb) cubemap.size() -- [6, 512, 512, 3]
    # (Pdb) cubemap.min() -- 0., cubemap.max() -- 109.8704 !!!
    l = EnvironmentLight(cubemap)
    l.build_mips()
    # l.env_base.size() -- [6, 512, 512, 3]
    return l

def load_env(fn, scale=1.0):
    if os.path.splitext(fn)[1].lower() == ".hdr":
        return _load_env_hdr(fn, scale)
    else:
        assert False, "Unknown envlight extension %s" % os.path.splitext(fn)[1]

def save_env_map(fn, light):
    # fn -- 'out/bob/dmtet_mesh/probe.hdr'
    assert isinstance(light, EnvironmentLight), "Can only save EnvironmentLight currently"
    if isinstance(light, EnvironmentLight): # True
        color = util.cubemap_to_latlong(light.base, [512, 1024])
    util.save_image_raw(fn, color.detach().cpu().numpy())

######################################################################################
# Create trainable env map with random initialization
######################################################################################

def create_trainable_env_rnd(base_res, scale=0.5, bias=0.25):
    # base_res = 512
    # scale = 0.0
    # bias = 0.5
    base = torch.rand(6, base_res, base_res, 3, dtype=torch.float32, device='cuda') * scale + bias
    # base.size() -- [6, 512, 512, 3]
    return EnvironmentLight(base)
      
