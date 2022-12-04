# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.

import torch
import nvdiffrast.torch as dr

from . import util
from . import renderutils as ru
from . import light

import pdb

# ==============================================================================================
#  Helper functions
# ==============================================================================================
def interpolate(attr, rast, attr_idx, rast_db=None):
    # attr.size() -- [1, 5344, 3]
    # rast.size() -- [1, 512, 512, 4]
    # attr_idx.size() -- [10688, 3]
    # rast_db = None
    return dr.interpolate(attr.contiguous(), rast, attr_idx, rast_db=rast_db, diff_attrs=None if rast_db is None else 'all')

# ==============================================================================================
#  pixel shader
# ==============================================================================================
def shade(
        gb_pos,
        gb_geometric_normal,
        gb_normal,
        gb_tangent,
        gb_texc,
        gb_texc_deriv,
        view_pos,
        lgt,
        material
    ):
    # gb_pos.size() -- [1, 512, 512, 3]
    # gb_geometric_normal.size() -- [1, 512, 512, 3]
    # gb_normal.size() -- [1, 512, 512, 3]
    # gb_tangent.size() -- [1, 512, 512, 3]
    # gb_texc.size() -- [1, 512, 512, 2]
    # gb_texc_deriv.size() -- [1, 512, 512, 4]

    # view_pos = tensor([[[[-0.4691, -0.7334,  2.7648]]]], device='cuda:0')
    # lgt = EnvironmentLight()
    # material = Material(
    #   (kd): Texture2D()
    #   (ks): Texture2D()
    # )

    ################################################################################
    # Texture lookups
    ################################################################################
    perturbed_nrm = None
    if 'kd_ks_normal' in material:
        # ==> Here !!!
        # Combined texture, used for MLPs because lookups are expensive
        all_tex_jitter = material['kd_ks_normal'].sample(gb_pos + torch.normal(mean=0, std=0.01, size=gb_pos.shape, device="cuda"))
        all_tex = material['kd_ks_normal'].sample(gb_pos)
        assert all_tex.shape[-1] == 9 or all_tex.shape[-1] == 10, "Combined kd_ks_normal must be 9 or 10 channels"
        kd, ks, perturbed_nrm = all_tex[..., :-6], all_tex[..., -6:-3], all_tex[..., -3:]
        # Compute albedo (kd) gradient, used for material regularizer
        kd_grad = torch.sum(torch.abs(all_tex_jitter[..., :-6] - all_tex[..., :-6]), dim=-1, keepdim=True) / 3
    else:
        # ==> pdb.set_trace()
        kd_jitter  = material['kd'].sample(gb_texc + torch.normal(mean=0, std=0.005, size=gb_texc.shape, device="cuda"), gb_texc_deriv)
        kd = material['kd'].sample(gb_texc, gb_texc_deriv)
        ks = material['ks'].sample(gb_texc, gb_texc_deriv)[..., 0:3] # skip alpha
        if 'normal' in material:
            # ==> pdb.set_trace
            perturbed_nrm = material['normal'].sample(gb_texc, gb_texc_deriv)
        kd_grad = torch.sum(torch.abs(kd_jitter[..., 0:3] - kd[..., 0:3]), dim=-1, keepdim=True) / 3

    # Separate kd into alpha and color, default alpha = 1
    alpha = kd[..., 3:4] if kd.shape[-1] == 4 else torch.ones_like(kd[..., 0:1]) # [1, 512, 512, 1]
    kd = kd[..., 0:3] # [1, 512, 512, 3]

    ################################################################################
    # Normal perturbation & normal bend
    ################################################################################
    gb_normal = ru.shading_normal(gb_pos, view_pos, perturbed_nrm, gb_normal, gb_tangent, gb_geometric_normal, two_sided_shading=True, opengl=True)

    ################################################################################
    # Evaluate BSDF
    ################################################################################
    if isinstance(lgt, light.EnvironmentLight):
        shaded_color = lgt.shade(gb_pos, gb_normal, kd, ks, view_pos, specular=True)
    else:
        assert False, "Invalid light type"

    # Return multiple buffers
    buffers = {
        'shaded'    : torch.cat((shaded_color, alpha), dim=-1), # [1, 512, 512, 4]
        'kd_grad'   : torch.cat((kd_grad, alpha), dim=-1), # [1, 512, 512, 2]
        'occlusion' : torch.cat((ks[..., :1], alpha), dim=-1) # [1, 512, 512, 2]
    }

    return buffers

# ==============================================================================================
#  Render a depth slice of the mesh (scene), some limitations:
#  - Single mesh
#  - Single light
#  - Single material
# ==============================================================================================
def render_layer(
        rast,
        rast_deriv,
        mesh,
        view_pos,
        lgt,
        resolution,
        spp,
        msaa,
    ):
    # rast.size() -- [1, 512, 512, 4]
    # rast_deriv.size() -- [1, 512, 512, 4]

    # mesh = <render.mesh.Mesh object at 0x7f2d55d87fa0>
    # view_pos = tensor([[[[ 2.3236, -0.2347,  1.7603]]]], device='cuda:0')
    # lgt = EnvironmentLight()
    # resolution = [512, 512]
    # spp ====== 1
    # msaa = True

    full_res = [resolution[0]*spp, resolution[1]*spp]

    ################################################################################
    # Rasterize
    ################################################################################

    # Scale down to shading resolution when MSAA is enabled, otherwise shade at full resolution
    if spp > 1 and msaa:
        rast_out_s = util.scale_img_nhwc(rast, resolution, mag='nearest', min='nearest')
        rast_out_deriv_s = util.scale_img_nhwc(rast_deriv, resolution, mag='nearest', min='nearest') * spp
    else:
        rast_out_s = rast
        rast_out_deriv_s = rast_deriv

    ################################################################################
    # Interpolate attributes
    ################################################################################

    # Interpolate world space position
    gb_pos, _ = interpolate(mesh.v_pos[None, ...], rast_out_s, mesh.t_pos_idx.int())

    # Compute geometric normals. We need those because of bent normals trick (for bump mapping)
    v0 = mesh.v_pos[mesh.t_pos_idx[:, 0], :]
    v1 = mesh.v_pos[mesh.t_pos_idx[:, 1], :]
    v2 = mesh.v_pos[mesh.t_pos_idx[:, 2], :]
    face_normals = util.safe_normalize(torch.cross(v1 - v0, v2 - v0))
    face_normal_indices = (torch.arange(0, face_normals.shape[0], dtype=torch.int64, device='cuda')[:, None]).repeat(1, 3)
    gb_geometric_normal, _ = interpolate(face_normals[None, ...], rast_out_s, face_normal_indices.int())

    # Compute tangent space
    assert mesh.v_nrm is not None and mesh.v_tng is not None
    gb_normal, _ = interpolate(mesh.v_nrm[None, ...], rast_out_s, mesh.t_nrm_idx.int())
    gb_tangent, _ = interpolate(mesh.v_tng[None, ...], rast_out_s, mesh.t_tng_idx.int()) # Interpolate tangents

    # Texture coordinate
    assert mesh.v_tex is not None
    gb_texc, gb_texc_deriv = interpolate(mesh.v_tex[None, ...], rast_out_s, mesh.t_tex_idx.int(), rast_db=rast_out_deriv_s)

    ################################################################################
    # Shade
    ################################################################################

    buffers = shade(gb_pos, gb_geometric_normal, gb_normal, gb_tangent, gb_texc, gb_texc_deriv, 
        view_pos, lgt, mesh.material)

    ################################################################################
    # Prepare output
    ################################################################################

    # Scale back up to visibility resolution if using MSAA
    if spp > 1 and msaa:
        for key in buffers.keys():
            buffers[key] = util.scale_img_nhwc(buffers[key], full_res, mag='nearest', min='nearest')

    # Return buffers
    return buffers

# ==============================================================================================
#  Render a depth peeled mesh (scene), some limitations:
#  - Single mesh
#  - Single light
#  - Single material
# ==============================================================================================
def render_mesh(
        ctx,
        mesh,
        mtx_in,
        view_pos,
        lgt,
        resolution,
        spp         = 1,
        num_layers  = 1,
        msaa        = False,
        background  = None, 
    ):
    # ctx = <nvdiffrast.torch.ops.RasterizeGLContext object at 0x7f88cfa59940>
    # mesh = <render.mesh.Mesh object at 0x7f88cf9e9be0>
    # mtx_in = tensor([[[-1.6353,  1.0194,  1.4543, -0.5719],
    #          [ 1.7332,  1.3474,  1.0045, -0.2639],
    #          [-0.1605,  0.7144, -0.6813,  2.8980],
    #          [-0.1605,  0.7143, -0.6812,  3.0974]]], device='cuda:0')
    # view_pos = tensor([[[[ 0.4151, -2.0515,  2.2981]]]], device='cuda:0')
    # lgt = EnvironmentLight()
    # resolution = [512, 512]
    # msaa = True
    # background.size() -- [1, 512, 512, 4]

    def prepare_input_vector(x):
        x = torch.tensor(x, dtype=torch.float32, device='cuda') if not torch.is_tensor(x) else x
        return x[:, None, None, :] if len(x.shape) == 2 else x
    
    def composite_buffer(key, layers, background, antialias):
        accum = background
        for buffers, rast in reversed(layers):
            alpha = (rast[..., -1:] > 0).float() * buffers[key][..., -1:]
            accum = torch.lerp(accum, torch.cat((buffers[key][..., :-1], torch.ones_like(buffers[key][..., -1:])), dim=-1), alpha)
            if antialias: # True for shaded
                accum = dr.antialias(accum.contiguous(), rast, v_pos_clip, mesh.t_pos_idx.int())
        return accum

    assert mesh.t_pos_idx.shape[0] > 0, "Got empty training triangle mesh (unrecoverable discontinuity)"
    assert background is None or (background.shape[1] == resolution[0] and background.shape[2] == resolution[1])

    full_res = [resolution[0]*spp, resolution[1]*spp]

    # Convert numpy arrays to torch tensors
    mtx_in      = torch.tensor(mtx_in, dtype=torch.float32, device='cuda') if not torch.is_tensor(mtx_in) else mtx_in
    view_pos    = prepare_input_vector(view_pos)

    # clip space transform
    v_pos_clip = ru.points_transform(mesh.v_pos[None, ...], mtx_in)

    # Render all layers front-to-back
    layers = []
    # v_pos_clip.size() -- [1, 5344, 4]
    # mesh.t_pos_idx.size() -- [10688, 3]
    # full_res -- [512, 512]
    # proto type: nvdiffrast.torch.DepthPeeler(glctx, pos, tri, resolution)
    with dr.DepthPeeler(ctx, v_pos_clip, mesh.t_pos_idx.int(), full_res) as peeler:
        for _ in range(num_layers): # num_layers -- 1
            rast, db = peeler.rasterize_next_layer()
            # rast.size() -- [1, 512, 512, 4]
            # db.size() -- [1, 512, 512, 4]
            # resolution, spp, msaa -- ([512, 512], 1, True)
            layers += [(render_layer(rast, db, mesh, view_pos, lgt, resolution, spp, msaa), rast)]

    # Setup background
    if background is not None: # False ?
        if spp > 1: # False for spp == 1
            background = util.scale_img_nhwc(background, full_res, mag='nearest', min='nearest')
        background = torch.cat((background, torch.zeros_like(background[..., 0:1])), dim=-1)
    else:
        background = torch.zeros(1, full_res[0], full_res[1], 4, dtype=torch.float32, device='cuda')

    # Composite layers front-to-back
    out_buffers = {}
    for key in layers[0][0].keys():
        # layers[0][0].keys() -- dict_keys(['shaded', 'kd_grad', 'occlusion'])
        if key == 'shaded':
            accum = composite_buffer(key, layers, background, True)
        else:
            accum = composite_buffer(key, layers, torch.zeros_like(layers[0][0][key]), False)

        # Downscale to framebuffer resolution. Use avg pooling 
        out_buffers[key] = util.avg_pool_nhwc(accum, spp) if spp > 1 else accum

    return out_buffers

# ==============================================================================================
#  Render UVs
# ==============================================================================================
def render_uv(ctx, mesh, resolution, mlp_texture):
    # ctx = <nvdiffrast.torch.ops.RasterizeGLContext object at 0x7f7b62bfea60>
    # mesh = <render.mesh.Mesh object at 0x7f7b480270d0>
    # resolution = [1024, 1024]
    # mlp_texture = Texture(
    #   (encoder): Encoding(n_input_dims=3, n_output_dims=32, seed=1337, dtype=torch.float16, hyperparams={'base_resolution': 16, 'interpolation': 'Linear', 'log2_hashmap_size': 19, 'n_features_per_level': 2, 'n_levels': 16, 'otype': 'Grid', 'per_level_scale': 1.4472692012786865, 'type': 'Hash'})
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

    # clip space transform 
    uv_clip = mesh.v_tex[None, ...]*2.0 - 1.0
    # uv_clip.size() -- [1, 7038, 2]

    # pad to four component coordinate
    uv_clip4 = torch.cat((uv_clip, torch.zeros_like(uv_clip[...,0:1]), torch.ones_like(uv_clip[...,0:1])), dim = -1)
    # uv_clip4.size() -- [1, 7038, 4]

    # rasterize
    rast, _ = dr.rasterize(ctx, uv_clip4, mesh.t_tex_idx.int(), resolution)

    # Interpolate world space position
    gb_pos, _ = interpolate(mesh.v_pos[None, ...], rast, mesh.t_pos_idx.int())
    # gb_pos.size() -- [1, 1024, 1024, 3]

    # Sample out textures from MLP
    all_tex = mlp_texture.sample(gb_pos)
    assert all_tex.shape[-1] == 9 or all_tex.shape[-1] == 10, "Combined kd_ks_normal must be 9 or 10 channels"
    perturbed_nrm = all_tex[..., -3:]

    # rast.size() -- [1, 1024, 1024, 4]
    # all_tex.size() -- [1, 1024, 1024, 9]
    # perturbed_nrm.size() -- [1, 1024, 1024, 3]
    # mask, kd, ks, normal
    return (rast[..., -1:] > 0).float(), all_tex[..., :-6], all_tex[..., -6:-3], util.safe_normalize(perturbed_nrm)
