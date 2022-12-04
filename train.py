# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.
import pdb

import os
import time
import argparse
import json

import numpy as np
import torch
import nvdiffrast.torch as dr
import xatlas

# Import data readers / generators
from dataset.dataset_mesh import DatasetMesh
from dataset.dataset_nerf import DatasetNERF
from dataset.dataset_llff import DatasetLLFF

# Import topology / geometry trainers
from geometry.dmtet import DMTetGeometry
from geometry.dlmesh import RefineMesh

from render import obj
from render import material
from render import util
from render import mesh
from render import texture
from render import mlptexture
from render import light
from render import render

RADIUS = 3.0

# Enable to debug back-prop anomalies
# torch.autograd.set_detect_anomaly(True)


###############################################################################
# Mix background into a dataset image
###############################################################################

@torch.no_grad()
def prepare_batch(target, bg_type='black'):
    assert bg_type in ["random", "white"]
    # print(f"prepare_batch: target shape -- {target['img'].shape}, bg_type -- {bg_type}")
    # target shape -- torch.Size([4, 512, 512, 4]), bg_type -- random/white
    assert len(target['img'].shape) == 4, "Image shape should be [n, h, w, c]"
    if bg_type == 'checker':
        background = torch.tensor(util.checkerboard(target['img'].shape[1:3], 8), dtype=torch.float32, device='cuda')[None, ...]
    elif bg_type == 'black':
        background = torch.zeros(target['img'].shape[0:3] + (3,), dtype=torch.float32, device='cuda')
    elif bg_type == 'white': # True
        background = torch.ones(target['img'].shape[0:3] + (3,), dtype=torch.float32, device='cuda')
    elif bg_type == 'reference':
        background = target['img'][..., 0:3]
    elif bg_type == 'random': # True
        background = torch.rand(target['img'].shape[0:3] + (3,), dtype=torch.float32, device='cuda')
    else:
        assert False, "Unknown background type %s" % bg_type

    target['mv'] = target['mv'].cuda()
    target['mvp'] = target['mvp'].cuda()
    target['campos'] = target['campos'].cuda()
    target['img'] = target['img'].cuda()
    target['background'] = background

    target['img'] = torch.cat((torch.lerp(background, target['img'][..., 0:3], target['img'][..., 3:4]), target['img'][..., 3:4]), dim=-1)

    return target

###############################################################################
# UV - map geometry & convert to a mesh
###############################################################################

@torch.no_grad()
def xatlas_uvmap(glctx, geometry, mat, FLAGS):
    # glctx = <nvdiffrast.torch.ops.RasterizeGLContext object at 0x7fe9225b3eb0>
    # geometry = DMTetGeometry()
    # mat = Material(
    #   (kd_ks_normal): Texture(
    #     (encoder): Encoding(n_input_dims=3, n_output_dims=32, seed=1337, dtype=torch.float16, hyperparams={'base_resolution': 16, 'interpolation': 'Linear', 'log2_hashmap_size': 19, 'n_features_per_level': 2, 'n_levels': 16, 'otype': 'Grid', 'per_level_scale': 1.4472692012786865, 'type': 'Hash'})
    #     (net): _MLP()
    #   )
    # )
    eval_mesh = geometry.getMesh(mat)
    
    # Create uvs with xatlas
    v_pos = eval_mesh.v_pos.detach().cpu().numpy()
    t_pos_idx = eval_mesh.t_pos_idx.detach().cpu().numpy()
    vmapping, indices, uvs = xatlas.parametrize(v_pos, t_pos_idx)

    # Convert to tensors
    indices_int64 = indices.astype(np.uint64, casting='same_kind').view(np.int64)
    
    uvs = torch.tensor(uvs, dtype=torch.float32, device='cuda')
    faces = torch.tensor(indices_int64, dtype=torch.int64, device='cuda')

    new_mesh = mesh.Mesh(v_tex=uvs, t_tex_idx=faces, base=eval_mesh)

    mask, kd, ks, normal = render.render_uv(glctx, new_mesh, FLAGS.texture_res, 
        eval_mesh.material['kd_ks_normal'])
    
    if FLAGS.layers > 1:
        kd = torch.cat((kd, torch.rand_like(kd[...,0:1])), dim=-1)

    kd_min, kd_max = torch.tensor(FLAGS.kd_min, dtype=torch.float32, device='cuda'), torch.tensor(FLAGS.kd_max, dtype=torch.float32, device='cuda')
    ks_min, ks_max = torch.tensor(FLAGS.ks_min, dtype=torch.float32, device='cuda'), torch.tensor(FLAGS.ks_max, dtype=torch.float32, device='cuda')
    nrm_min, nrm_max = torch.tensor(FLAGS.nrm_min, dtype=torch.float32, device='cuda'), torch.tensor(FLAGS.nrm_max, dtype=torch.float32, device='cuda')

    new_mesh.material = material.Material({
        'bsdf'   : mat['bsdf'],
        'kd'     : texture.Texture2D(kd, min_max=[kd_min, kd_max]),
        'ks'     : texture.Texture2D(ks, min_max=[ks_min, ks_max]),
        'normal' : texture.Texture2D(normal, min_max=[nrm_min, nrm_max])
    })

    return new_mesh

###############################################################################
# Utility functions for material
###############################################################################

def initial_guess_material(geometry, FLAGS):
    # mlp = True

    kd_min, kd_max = torch.tensor(FLAGS.kd_min, dtype=torch.float32, device='cuda'), torch.tensor(FLAGS.kd_max, dtype=torch.float32, device='cuda')
    ks_min, ks_max = torch.tensor(FLAGS.ks_min, dtype=torch.float32, device='cuda'), torch.tensor(FLAGS.ks_max, dtype=torch.float32, device='cuda')
    nrm_min, nrm_max = torch.tensor(FLAGS.nrm_min, dtype=torch.float32, device='cuda'), torch.tensor(FLAGS.nrm_max, dtype=torch.float32, device='cuda')

    # kd_min [0.0, 0.0, 0.0, 0.0]
    # kd_max [1.0, 1.0, 1.0, 1.0]
    # ks_min [0, 0.25, 0]
    # ks_max [1.0, 1.0, 1.0]
    # nrm_min [-1.0, -1.0, 0.0]
    # nrm_max [1.0, 1.0, 1.0]

    mlp_min = torch.cat((kd_min[0:3], ks_min, nrm_min), dim=0) # mlp_min.size() -- [9], kd, ks, normal
    mlp_max = torch.cat((kd_max[0:3], ks_max, nrm_max), dim=0) # mlp_max.size() -- [9], kd, ks, normal

    # geometry.getAABB()
    # tensor([-1.0500, -1.0500, -1.0500], device='cuda:0'), 
    # tensor([1.0500, 1.0500, 1.0500], device='cuda:0')
    mlp_map_opt = mlptexture.Texture(geometry.getAABB(), channels=9, min_max=[mlp_min, mlp_max])

    # mlp_map_opt
    # Texture(
    #   (encoder): Encoding(n_input_dims=3, n_output_dims=32, seed=1337, 
    #         dtype=torch.float16, hyperparams={'base_resolution': 16, 
    #         'interpolation': 'Linear', 'log2_hashmap_size': 19, 
    #         'n_features_per_level': 2, 'n_levels': 16, 'otype': 'Grid', 
    #         'per_level_scale': 1.4472692012786865, 'type': 'Hash'})
    #   (net): _MLP(
    #     (net): Sequential(
    #       (0): Linear(in_features=32, out_features=32, bias=False)
    #       (1): ReLU()
    #       (2): Linear(in_features=32, out_features=32, bias=False)
    #       (3): ReLU()
    #       (4): Linear(in_features=32, out_features=9, bias=False)
    #     )
    #   )
    # )
    mat =  material.Material({'kd_ks_normal' : mlp_map_opt})
    mat['bsdf'] = 'pbr'

    return mat

###############################################################################
# Validation & testing
###############################################################################

def validate_itr(glctx, target, geometry, opt_material, lgt, FLAGS):
    result_dict = {}
    with torch.no_grad():
        lgt.build_mips()

        buffers = geometry.render(glctx, target, lgt, opt_material)

        result_dict['ref'] = util.rgb_to_srgb(target['img'][...,0:3])[0]
        result_dict['opt'] = util.rgb_to_srgb(buffers['shaded'][...,0:3])[0]
        result_image = torch.cat([result_dict['opt'], result_dict['ref']], axis=1)
        return result_image, result_dict

def validate(glctx, geometry, opt_material, lgt, dataset_validate, out_dir, FLAGS):
    # ==============================================================================================
    #  Validation loop
    # ==============================================================================================
    mse_values = []
    psnr_values = []

    dataloader_validate = torch.utils.data.DataLoader(dataset_validate, batch_size=1, collate_fn=dataset_validate.collate)

    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, 'metrics.txt'), 'w') as fout:
        fout.write('ID, MSE, PSNR\n')

        print("Running validation")
        for it, target in enumerate(dataloader_validate):

            # Mix validation background
            target = prepare_batch(target, FLAGS.background)

            result_image, result_dict = validate_itr(glctx, target, geometry, opt_material, lgt, FLAGS)
           
            # Compute metrics
            opt = torch.clamp(result_dict['opt'], 0.0, 1.0) 
            ref = torch.clamp(result_dict['ref'], 0.0, 1.0)

            mse = torch.nn.functional.mse_loss(opt, ref, size_average=None, reduce=None, reduction='mean').item()
            mse_values.append(float(mse))
            psnr = util.mse_to_psnr(mse)
            psnr_values.append(float(psnr))

            line = "%d, %1.8f, %1.8f\n" % (it, mse, psnr)
            fout.write(str(line))

            for k in result_dict.keys():
                np_img = result_dict[k].detach().cpu().numpy()
                util.save_image(out_dir + '/' + ('val_%06d_%s.png' % (it, k)), np_img)

        avg_mse = np.mean(np.array(mse_values))
        avg_psnr = np.mean(np.array(psnr_values))
        line = "AVERAGES: %1.4f, %2.3f\n" % (avg_mse, avg_psnr)
        fout.write(str(line))
        print("MSE,      PSNR")
        print("%1.8f, %2.3f" % (avg_mse, avg_psnr))
    return avg_psnr

###############################################################################
# Main shape fitter function / optimization loop
###############################################################################

class Trainer(torch.nn.Module):
    def __init__(self, glctx, geometry, lgt, mat, image_loss_fn, FLAGS):
        super(Trainer, self).__init__()
        # geometry = DMTetGeometry()
        # lgt = EnvironmentLight()
        # mat = Material(
        #   (kd_ks_normal): Texture(
        #     (encoder): Encoding(n_input_dims=3, n_output_dims=32, seed=1337, dtype=torch.float16, hyperparams={'base_resolution': 16, 'interpolation': 'Linear', 'log2_hashmap_size': 19, 'n_features_per_level': 2, 'n_levels': 16, 'otype': 'Grid', 'per_level_scale': 1.4472692012786865, 'type': 'Hash'})
        #     (net): _MLP(
        #       (net): Sequential(
        #         (0): Linear(in_features=32, out_features=32, bias=False)
        #         (1): ReLU()
        #         (2): Linear(in_features=32, out_features=32, bias=False)
        #         (3): ReLU()
        #         (4): Linear(in_features=32, out_features=9, bias=False)
        #       )
        #     )
        #   )
        # )
        self.glctx = glctx
        self.geometry = geometry
        self.light = lgt
        self.material = mat
        self.image_loss_fn = image_loss_fn
        self.FLAGS = FLAGS

    def forward(self, target, it):
        self.light.build_mips()
        return self.geometry.tick(glctx, target, self.light, self.material, self.image_loss_fn, it)

def optimize_mesh(
    glctx,
    geometry,
    opt_material,
    lgt,
    dataset_train,
    dataset_validate,
    FLAGS,
    warmup_iter=0,
    log_interval=10,
    pass_idx=0,
    pass_name="",
    ):
    # glctx = <nvdiffrast.torch.ops.RasterizeGLContext object at 0x7fb240aad580>
    # geometry = DMTetGeometry()
    # opt_material = Material(
    #   (kd_ks_normal): Texture(
    #     (encoder): Encoding(n_input_dims=3, n_output_dims=32, seed=1337, dtype=torch.float16, hyperparams={'base_resolution': 16, 'interpolation': 'Linear', 'log2_hashmap_size': 19, 'n_features_per_level': 2, 'n_levels': 16, 'otype': 'Grid', 'per_level_scale': 1.4472692012786865, 'type': 'Hash'})
    #     (net): _MLP(
    #       (net): Sequential(
    #         (0): Linear(in_features=32, out_features=32, bias=False)
    #         (1): ReLU()
    #         (2): Linear(in_features=32, out_features=32, bias=False)
    #         (3): ReLU()
    #         (4): Linear(in_features=32, out_features=9, bias=False)
    #       )
    #     )
    #   )
    # )
    # lgt = EnvironmentLight()
    # dataset_train = <dataset.dataset_mesh.DatasetMesh object at 0x7fb240aadee0>
    # dataset_validate = <dataset.dataset_mesh.DatasetMesh object at 0x7fb240a89b50>
    # FLAGS = Namespace(config='configs/bob.json', iter=200, batch=4, spp=1, layers=1, 
    # train_res=[512, 512], texture_res=[1024, 1024], display_interval=0, save_interval=100, 
    # learning_rate=[0.03, 0.003], min_roughness=0.08,background='white', loss='logl1', 
    # out_dir='out/bob', ref_mesh='data/bob/bob_tri.obj', base_mesh=None, validate=False, 
    # mtl_override=None, dmtet_grid=64, mesh_scale=2.1, env_scale=2.0, 
    # envmap='data/irrmaps/aerodynamics_workshop_2k.hdr', display=None, sdf_regularizer=0.2, 
    # laplace='relative', laplace_scale=10000.0, pre_load=True, kd_min=[0.0, 0.0, 0.0, 0.0], 
    # kd_max=[1.0, 1.0, 1.0, 1.0], ks_min=[0, 0.25, 0], ks_max=[1.0, 1.0, 1.0], 
    # nrm_min=[-1.0, -1.0, 0.0], nrm_max=[1.0, 1.0, 1.0], cam_near_far=[0.1, 1000.0])
    # pass_name = 'dmtet_pass1'

    # ==============================================================================================
    #  Setup torch optimizer
    # ==============================================================================================

    learning_rate = FLAGS.learning_rate[pass_idx] if isinstance(FLAGS.learning_rate, list) or isinstance(FLAGS.learning_rate, tuple) else FLAGS.learning_rate
    learning_rate_pos = learning_rate[0] if isinstance(learning_rate, list) or isinstance(learning_rate, tuple) else learning_rate
    learning_rate_mat = learning_rate[1] if isinstance(learning_rate, list) or isinstance(learning_rate, tuple) else learning_rate

    def lr_schedule(iter, fraction):
        if iter < warmup_iter:
            return iter / warmup_iter 
        return max(0.0, 10**(-(iter - warmup_iter)*0.0002)) # Exponential falloff from [1.0, 0.1] over 5k epochs.    

    # ==============================================================================================
    #  Image loss
    # ==============================================================================================
    image_loss_fn = torch.nn.functional.l1_loss # createLoss(FLAGS)
    trainer = Trainer(glctx, geometry, lgt, opt_material, image_loss_fn, FLAGS)

    # Single GPU training mode
    optimizer_mesh = torch.optim.Adam(list(trainer.geometry.parameters()), lr=learning_rate_pos)
    # for k in trainer.geometry.parameters(): print(k.size())
    # [36562] -- sdf
    # [36562, 3] -- deform

    scheduler_mesh = torch.optim.lr_scheduler.LambdaLR(optimizer_mesh, lr_lambda=lambda x: lr_schedule(x, 0.9)) 
    # (Pdb) optimizer_mesh
    # Adam (
    # Parameter Group 0
    #     amsgrad: False
    #     betas: (0.9, 0.999)
    #     eps: 1e-08
    #     initial_lr: 0.03
    #     lr: 0.03
    #     weight_decay: 0
    # )

    optimizer = torch.optim.Adam(list(trainer.material.parameters()) + list(trainer.light.parameters()), lr=learning_rate_mat)
    # for k in trainer.material.parameters(): print(k.size())
    # [12599920] -- encoder
    # [32, 32],  [32, 32], [9, 32] -- MLP network
    # for k in trainer.light.parameters(): print(k.size())
    # [6, 512, 512, 3] -- base, cube map

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: lr_schedule(x, 0.9)) 
    # (Pdb) optimizer
    # Adam (
    # Parameter Group 0
    #     amsgrad: False
    #     betas: (0.9, 0.999)
    #     eps: 1e-08
    #     initial_lr: 0.03
    #     lr: 0.03
    #     weight_decay: 0
    # )

    # ==============================================================================================
    #  Training loop
    # ==============================================================================================
    img_cnt = 0
    img_loss_vec = []
    reg_loss_vec = []
    iter_dur_vec = []

    dataloader_train    = torch.utils.data.DataLoader(dataset_train, batch_size=FLAGS.batch, collate_fn=dataset_train.collate, shuffle=True)
    dataloader_validate = torch.utils.data.DataLoader(dataset_validate, batch_size=1, collate_fn=dataset_train.collate)

    def cycle(iterable):
        iterator = iter(iterable)
        while True:
            try:
                yield next(iterator)
            except StopIteration:
                iterator = iter(iterable)

    v_it = cycle(dataloader_validate)

    for it, target in enumerate(dataloader_train):

        # Mix randomized background into dataset image
        target = prepare_batch(target, 'random')

        # ==============================================================================================
        #  Display / save outputs. Do it before training so we get initial meshes
        # ==============================================================================================

        # Show/save image before training step (want to get correct rendering of input)
        display_image = FLAGS.display_interval and (it % FLAGS.display_interval == 0)
        save_image = FLAGS.save_interval and (it % FLAGS.save_interval == 0)
        if display_image or save_image:
            result_image, result_dict = validate_itr(glctx, prepare_batch(next(v_it), FLAGS.background), geometry, opt_material, lgt, FLAGS)
            np_result_image = result_image.detach().cpu().numpy()
            if display_image:
                util.display_image(np_result_image, title='%d / %d' % (it, FLAGS.iter))
            if save_image:
                util.save_image(FLAGS.out_dir + '/' + ('img_%s_%06d.png' % (pass_name, img_cnt)), np_result_image)
                img_cnt = img_cnt+1

        start_time = time.time()

        # ==============================================================================================
        #  Zero gradients
        # ==============================================================================================
        optimizer.zero_grad()
        optimizer_mesh.zero_grad()

        # ==============================================================================================
        #  Training
        # ==============================================================================================
        img_loss, reg_loss = trainer(target, it)

        # ==============================================================================================
        #  Final loss
        # ==============================================================================================
        total_loss = img_loss + reg_loss

        img_loss_vec.append(img_loss.item())
        reg_loss_vec.append(reg_loss.item())

        # ==============================================================================================
        #  Backpropagate
        # ==============================================================================================
        total_loss.backward()
        if hasattr(lgt, 'base') and lgt.base.grad is not None: # True
            lgt.base.grad *= 64
        if 'kd_ks_normal' in opt_material: # True
            opt_material['kd_ks_normal'].encoder.params.grad /= 8.0

        optimizer.step()
        scheduler.step()

        optimizer_mesh.step()
        scheduler_mesh.step()

        # ==============================================================================================
        #  Clamp trainables to reasonable range
        # ==============================================================================================
        with torch.no_grad():
            if 'kd' in opt_material: # False
                opt_material['kd'].clamp_()
            if 'ks' in opt_material: # False
                opt_material['ks'].clamp_()
            if 'normal' in opt_material: # False
                opt_material['normal'].clamp_()
                opt_material['normal'].normalize_()
            if lgt is not None: # True
                lgt.clamp_(min=0.0)

        torch.cuda.current_stream().synchronize()
        iter_dur_vec.append(time.time() - start_time)

        # ==============================================================================================
        #  Logging
        # ==============================================================================================
        if it % log_interval == 0:
            img_loss_avg = np.mean(np.asarray(img_loss_vec[-log_interval:]))
            reg_loss_avg = np.mean(np.asarray(reg_loss_vec[-log_interval:]))
            iter_dur_avg = np.mean(np.asarray(iter_dur_vec[-log_interval:]))
            
            remaining_time = (FLAGS.iter-it)*iter_dur_avg
            print("iter=%5d, img_loss=%.6f, reg_loss=%.6f, lr=%.5f, time=%.1f ms, rem=%s" % 
                (it, img_loss_avg, reg_loss_avg, optimizer.param_groups[0]['lr'], iter_dur_avg*1000, util.time_to_text(remaining_time)))

    return geometry, opt_material

#----------------------------------------------------------------------------
# Main function.
#----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='nvdiffrec')
    parser.add_argument('--config', type=str, default=None, help='Config file')
    parser.add_argument('-i', '--iter', type=int, default=5000)
    parser.add_argument('-b', '--batch', type=int, default=1)
    parser.add_argument('-s', '--spp', type=int, default=1)
    parser.add_argument('-l', '--layers', type=int, default=1)
    parser.add_argument('-r', '--train-res', nargs=2, type=int, default=[512, 512])
    parser.add_argument('-dr', '--display-res', type=int, default=None)
    parser.add_argument('-tr', '--texture-res', nargs=2, type=int, default=[1024, 1024])
    parser.add_argument('-di', '--display-interval', type=int, default=0)
    parser.add_argument('-si', '--save-interval', type=int, default=1000)
    parser.add_argument('-lr', '--learning-rate', type=float, default=0.01)
    parser.add_argument('-mr', '--min-roughness', type=float, default=0.08)
    parser.add_argument('-mip', '--custom-mip', action='store_true', default=False)
    parser.add_argument('-rt', '--random-textures', action='store_true', default=False)
    parser.add_argument('-bg', '--background', default='checker', choices=['black', 'white', 'checker', 'reference'])
    parser.add_argument('--loss', default='logl1', choices=['logl1', 'logl2', 'mse', 'smape', 'relmse'])
    parser.add_argument('-o', '--out-dir', type=str, default=None)
    parser.add_argument('-rm', '--ref_mesh', type=str)
    parser.add_argument('-bm', '--base-mesh', type=str, default=None)
    parser.add_argument('--validate', type=bool, default=True)
    
    FLAGS = parser.parse_args()

    FLAGS.mtl_override        = None                     # Override material of model
    FLAGS.dmtet_grid          = 64                       # Resolution of initial tet grid. We provide 64 and 128 resolution grids. Other resolutions can be generated with https://github.com/crawforddoran/quartet
    FLAGS.mesh_scale          = 2.1                      # Scale of tet grid box. Adjust to cover the model
    FLAGS.env_scale           = 1.0                      # Env map intensity multiplier
    FLAGS.envmap              = None                     # HDR environment probe
    FLAGS.display             = None                     # Conf validation window/display. E.g. [{"relight" : <path to envlight>}]
    FLAGS.sdf_regularizer     = 0.2                      # Weight for sdf regularizer (see paper for details)
    FLAGS.laplace             = "relative"               # Mesh Laplacian ["absolute", "relative"]
    FLAGS.laplace_scale       = 10000.0                  # Weight for sdf regularizer. Default is relative with large weight
    FLAGS.pre_load            = True                     # Pre-load entire dataset into memory for faster training
    FLAGS.kd_min              = [ 0.0,  0.0,  0.0,  0.0] # Limits for kd
    FLAGS.kd_max              = [ 1.0,  1.0,  1.0,  1.0]
    FLAGS.ks_min              = [ 0.0, 0.08,  0.0]       # Limits for ks
    FLAGS.ks_max              = [ 1.0,  1.0,  1.0]
    FLAGS.nrm_min             = [-1.0, -1.0,  0.0]       # Limits for normal map
    FLAGS.nrm_max             = [ 1.0,  1.0,  1.0]
    FLAGS.cam_near_far        = [0.1, 1000.0]
    FLAGS.display_res         = [512, 512]
    
    # FLAGS --
    # Namespace(
    #     config='configs/bob.json', iter=200, batch=4, spp=1, layers=1, 
    #     train_res=[512, 512], display_res=[512, 512], texture_res=[1024, 1024], 
    #     display_interval=0, save_interval=100, learning_rate=[0.03, 0.003], 
    #     min_roughness=0.08,  
    #     background='white', loss='logl1', out_dir='out/bob', 
    #     ref_mesh='data/bob/bob_tri.obj', 
    #     base_mesh=None, validate=False, mtl_override=None, dmtet_grid=64, mesh_scale=2.1, 
    #     env_scale=2.0, envmap='data/irrmaps/aerodynamics_workshop_2k.hdr', display=None, 
    #     sdf_regularizer=0.2, laplace='relative', laplace_scale=10000.0, 
    #     pre_load=True, 
    #     kd_min=[0.0, 0.0, 0.0, 0.0], kd_max=[1.0, 1.0, 1.0, 1.0], 
    #     ks_min=[0, 0.25, 0], ks_max=[1.0, 1.0, 1.0], 
    #     nrm_min=[-1.0, -1.0, 0.0], nrm_max=[1.0, 1.0, 1.0], 
    #     cam_near_far=[0.1, 1000.0])

    if FLAGS.config is not None: # True
        data = json.load(open(FLAGS.config, 'r'))
        for key in data:
            FLAGS.__dict__[key] = data[key]

    FLAGS.out_dir = 'out/' + FLAGS.out_dir

    print("Config / Flags:")
    for key in FLAGS.__dict__.keys():
        print(key, FLAGS.__dict__[key])
    print("---------")

    os.makedirs(FLAGS.out_dir, exist_ok=True)

    glctx = dr.RasterizeGLContext() # <nvdiffrast.torch.ops.RasterizeGLContext object at 0x7fc2b5b39fd0> 

    # ==============================================================================================
    #  Create data pipeline
    # ==============================================================================================
    # FLAGS.ref_mesh -- 'data/bob/bob_tri.obj'
    if os.path.splitext(FLAGS.ref_mesh)[1] == '.obj':
        ref_mesh         = mesh.load_mesh(FLAGS.ref_mesh, FLAGS.mtl_override)
        dataset_train    = DatasetMesh(ref_mesh, glctx, RADIUS, FLAGS, validate=False)
        dataset_validate = DatasetMesh(ref_mesh, glctx, RADIUS, FLAGS, validate=True)
    elif os.path.isdir(FLAGS.ref_mesh):
        if os.path.isfile(os.path.join(FLAGS.ref_mesh, 'poses_bounds.npy')):
            dataset_train    = DatasetLLFF(FLAGS.ref_mesh, FLAGS, examples=(FLAGS.iter+1)*FLAGS.batch)
            dataset_validate = DatasetLLFF(FLAGS.ref_mesh, FLAGS)
        elif os.path.isfile(os.path.join(FLAGS.ref_mesh, 'transforms_train.json')):
            dataset_train    = DatasetNERF(os.path.join(FLAGS.ref_mesh, 'transforms_train.json'), FLAGS, examples=(FLAGS.iter+1)*FLAGS.batch)
            dataset_validate = DatasetNERF(os.path.join(FLAGS.ref_mesh, 'transforms_test.json'), FLAGS)

    # ==============================================================================================
    #  Create env light with trainable parameters
    # ==============================================================================================
    lgt = light.create_trainable_env_rnd(512, scale=0.0, bias=0.5)

    # ==============================================================================================
    #  If no initial guess, use DMtets to create geometry
    # ==============================================================================================
    # Setup geometry for optimization
    geometry = DMTetGeometry(FLAGS.dmtet_grid, FLAGS.mesh_scale, FLAGS) # geometry.dmtet.DMTetGeometry
    # Setup textures, make initial guess from reference if possible
    mat = initial_guess_material(geometry, FLAGS) # render.material.Material
    # Run optimization
    geometry, mat = optimize_mesh(glctx, geometry, mat, lgt, dataset_train, dataset_validate, 
                    FLAGS, pass_idx=0, pass_name="dmtet_pass1")

    if FLAGS.validate:
        validate(glctx, geometry, mat, lgt, dataset_validate, os.path.join(FLAGS.out_dir, "dmtet_validate"), FLAGS)

    # Create textured mesh from result
    base_mesh = xatlas_uvmap(glctx, geometry, mat, FLAGS) # render.mesh.Mesh

    # Free temporaries / cached memory 
    torch.cuda.empty_cache()
    mat['kd_ks_normal'].cleanup()
    del mat['kd_ks_normal']

    # Dump mesh for debugging.
    os.makedirs(os.path.join(FLAGS.out_dir, "dmtet_mesh"), exist_ok=True)
    obj.write_obj(os.path.join(FLAGS.out_dir, "dmtet_mesh/"), base_mesh)
    light.save_env_map(os.path.join(FLAGS.out_dir, "dmtet_mesh/probe.hdr"), lgt)

    # ==============================================================================================
    #  Pass 2: Train with fixed topology (mesh)
    # ==============================================================================================
    # lgt = lgt.clone() # need ?
    dlmesh = RefineMesh(base_mesh, FLAGS) # geometry.dlmesh.RefineMesh
    dlmesh, mat = optimize_mesh(glctx, dlmesh, base_mesh.material, lgt, dataset_train, dataset_validate, FLAGS, 
                pass_idx=1, pass_name="mesh_pass2", warmup_iter=100)


    # ==============================================================================================
    #  Validate
    # ==============================================================================================
    if FLAGS.validate:
        validate(glctx, dlmesh, mat, lgt, dataset_validate, os.path.join(FLAGS.out_dir, "validate"), FLAGS)

    # ==============================================================================================
    #  Dump output
    # ==============================================================================================
    final_mesh = dlmesh.getMesh(mat) # render.mesh.Mesh

    os.makedirs(os.path.join(FLAGS.out_dir, "mesh"), exist_ok=True)
    obj.write_obj(os.path.join(FLAGS.out_dir, "mesh/"), final_mesh)
    light.save_env_map(os.path.join(FLAGS.out_dir, "mesh/probe.hdr"), lgt)
