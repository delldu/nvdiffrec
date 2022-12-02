# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.

import torch
import torch.nn as nn

import torch.nn.functional as F
from render import mesh
from render import render
from render import regularizer
import pdb

###############################################################################
#  Geometry interface
###############################################################################

class DLMesh(nn.Module):
    def __init__(self, initial_guess, FLAGS):
        super(DLMesh, self).__init__()
        self.FLAGS = FLAGS
        self.initial_guess = initial_guess
        self.mesh = initial_guess.clone()
        print("Base mesh has %d triangles and %d vertices." % (self.mesh.t_pos_idx.shape[0], self.mesh.v_pos.shape[0]))
        
        self.mesh.v_pos = nn.Parameter(self.mesh.v_pos, requires_grad=True)
        self.register_parameter('vertex_pos', self.mesh.v_pos) # could not been detected !!!
        
    @torch.no_grad()
    def getAABB(self):
        pdb.set_trace()
        return mesh.aabb(self.mesh)

    def getMesh(self, material):
        self.mesh.material = material

        imesh = mesh.Mesh(base=self.mesh)
        # Compute normals and tangent space
        imesh = mesh.auto_normals(imesh)
        imesh = mesh.compute_tangents(imesh)
        return imesh

    def render(self, glctx, target, lgt, opt_material, bsdf=None):
        opt_mesh = self.getMesh(opt_material)
        # self.FLAGS.layers -- 1
        # (Pdb) opt_mesh.v_pos.size() -- [5238, 3]
        # (Pdb) opt_mesh.v_nrm.size() -- [5238, 3]
        # (Pdb) opt_mesh.v_tng.size() -- [5238, 3]
        # (Pdb) opt_mesh.v_tex.size() -- [7023, 2]

        # (Pdb) opt_mesh.t_pos_idx.size() -- [10472, 3]
        # (Pdb) opt_mesh.t_tex_idx.size() -- [10472, 3]
        # (Pdb) opt_mesh.t_tng_idx.size() -- [10472, 3]
        # (Pdb) opt_mesh.t_nrm_idx.size() -- [10472, 3]

        # target['mvp'].size() -- [1, 4, 4]
        # (Pdb) target['mvp']
        # tensor([[[ 2.4142,  0.0000,  0.0000,  0.0000],
        #          [ 0.0000, -2.2236,  0.9401,  0.0000],
        #          [ 0.0000, -0.3895, -0.9212,  2.8006],
        #          [ 0.0000, -0.3894, -0.9211,  3.0000]]], device='cuda:0')

        # (Pdb) target['campos'].size() -- [1, 3]
        # (Pdb) target['campos']
        # tensor([[-0.0000, 1.1683, 2.7632]], device='cuda:0')

        # target['resolution'] -- [512, 512]
        # (Pdb) target['background'].size() -- [1, 512, 512, 3]
        # (Pdb) target['background'].min(), max() -- 1.0, 1.0

        return render.render_mesh(glctx, opt_mesh, target['mvp'], target['campos'], lgt, target['resolution'], spp=target['spp'], 
                                    num_layers=self.FLAGS.layers, msaa=True, background=target['background'], bsdf=bsdf)

    def tick(self, glctx, target, lgt, opt_material, loss_fn, iteration):
        # ==============================================================================================
        #  Render optimizable object with identical conditions
        # ==============================================================================================
        buffers = self.render(glctx, target, lgt, opt_material)

        # ==============================================================================================
        #  Compute loss
        # ==============================================================================================
        t_iter = iteration / self.FLAGS.iter

        # Image-space loss, split into a coverage component and a color component
        color_ref = target['img']
        img_loss = F.mse_loss(buffers['shaded'][..., 3:], color_ref[..., 3:]) 
        img_loss += loss_fn(buffers['shaded'][..., 0:3] * color_ref[..., 3:], color_ref[..., 0:3] * color_ref[..., 3:])

        reg_loss = torch.tensor([0], dtype=torch.float32, device="cuda")

        # Compute regularizer.
        # self.FLAGS.laplace -- 'relative' 
        if self.FLAGS.laplace == "absolute":
            reg_loss += regularizer.laplace_regularizer_const(self.mesh.v_pos, self.mesh.t_pos_idx) * self.FLAGS.laplace_scale * (1 - t_iter)
        elif self.FLAGS.laplace == "relative": # True
            reg_loss += regularizer.laplace_regularizer_const(self.mesh.v_pos - self.initial_guess.v_pos, self.mesh.t_pos_idx) * self.FLAGS.laplace_scale * (1 - t_iter)                

        # Albedo (k_d) smoothnesss regularizer
        reg_loss += torch.mean(buffers['kd_grad'][..., :-1] * buffers['kd_grad'][..., -1:]) * 0.03 * min(1.0, iteration / 500)

        # Visibility regularizer
        reg_loss += torch.mean(buffers['occlusion'][..., :-1] * buffers['occlusion'][..., -1:]) * 0.001 * min(1.0, iteration / 500)

        # Light white balance regularizer
        reg_loss += lgt.regularizer() * 0.005

        return img_loss, reg_loss
