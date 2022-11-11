# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.
import pdb
import numpy as np
import torch

from render import util
from render import mesh
from render import render
from render import light

from .dataset import Dataset

###############################################################################
# Reference dataset using mesh & rendering
###############################################################################

class DatasetMesh(Dataset):

    def __init__(self, ref_mesh, glctx, cam_radius, FLAGS, validate=False):
        # Init 
        # self = <dataset.dataset_mesh.DatasetMesh object at 0x7fb240aadee0>
        # ref_mesh = <render.mesh.Mesh object at 0x7fb22855c640>
        # glctx = <nvdiffrast.torch.ops.RasterizeGLContext object at 0x7fb240aad580>
        # cam_radius = 3.0
        # FLAGS = Namespace(config='configs/bob.json', iter=200, batch=4, spp=1, layers=1, train_res=[512, 512], display_res=[512, 512], texture_res=[1024, 1024], display_interval=0, save_interval=100, learning_rate=[0.03, 0.003], min_roughness=0.08, custom_mip=False, random_textures=True, background='white', loss='logl1', out_dir='out/bob', ref_mesh='data/bob/bob_tri.obj', base_mesh=None, validate=False, mtl_override=None, dmtet_grid=64, mesh_scale=2.1, env_scale=2.0, envmap='data/irrmaps/aerodynamics_workshop_2k.hdr', display=None, camera_space_light=False, lock_light=False, lock_pos=False, sdf_regularizer=0.2, laplace='relative', laplace_scale=10000.0, pre_load=True, kd_min=[0.0, 0.0, 0.0, 0.0], kd_max=[1.0, 1.0, 1.0, 1.0], ks_min=[0, 0.25, 0], ks_max=[1.0, 1.0, 1.0], nrm_min=[-1.0, -1.0, 0.0], nrm_max=[1.0, 1.0, 1.0], cam_near_far=[0.1, 1000.0], learn_light=True, local_rank=0, multi_gpu=False)
        # validate = False

        self.glctx              = glctx
        self.cam_radius         = cam_radius
        self.FLAGS              = FLAGS
        self.validate           = validate
        self.fovy               = np.deg2rad(45) # 0.7853981633974483
        self.aspect             = FLAGS.train_res[1] / FLAGS.train_res[0]

        if self.FLAGS.local_rank == 0:
            print("DatasetMesh: ref mesh has %d triangles and %d vertices" % (ref_mesh.t_pos_idx.shape[0], ref_mesh.v_pos.shape[0]))

        # Sanity test training texture resolution
        ref_texture_res = np.maximum(ref_mesh.material['kd'].getRes(), ref_mesh.material['ks'].getRes())
        if 'normal' in ref_mesh.material:
            ref_texture_res = np.maximum(ref_texture_res, ref_mesh.material['normal'].getRes())
        if self.FLAGS.local_rank == 0 and FLAGS.texture_res[0] < ref_texture_res[0] or FLAGS.texture_res[1] < ref_texture_res[1]:
            print("---> WARNING: Picked a texture resolution lower than the reference mesh [%d, %d] < [%d, %d]" % (FLAGS.texture_res[0], FLAGS.texture_res[1], ref_texture_res[0], ref_texture_res[1]))

        # Load environment map texture
        self.envlight = light.load_env(FLAGS.envmap, scale=FLAGS.env_scale)
    
        self.ref_mesh = mesh.compute_tangents(ref_mesh)

    def _rotate_scene(self, itr):
        proj_mtx = util.perspective(self.fovy, self.FLAGS.display_res[1] / self.FLAGS.display_res[0], self.FLAGS.cam_near_far[0], self.FLAGS.cam_near_far[1])

        # Smooth rotation for display.
        ang    = (itr / 50) * np.pi * 2
        mv     = util.translate(0, 0, -self.cam_radius) @ (util.rotate_x(-0.4) @ util.rotate_y(ang))
        mvp    = proj_mtx @ mv
        campos = torch.linalg.inv(mv)[:3, 3]

        return mv[None, ...].cuda(), mvp[None, ...].cuda(), campos[None, ...].cuda(), self.FLAGS.display_res, self.FLAGS.spp

    def _random_scene(self):
        # ==============================================================================================
        #  Setup projection matrix
        # ==============================================================================================
        iter_res = self.FLAGS.train_res
        proj_mtx = util.perspective(self.fovy, iter_res[1] / iter_res[0], self.FLAGS.cam_near_far[0], self.FLAGS.cam_near_far[1])

        # ==============================================================================================
        #  Random camera & light position
        # ==============================================================================================

        # Random rotation/translation matrix for optimization.
        mv     = util.translate(0, 0, -self.cam_radius) @ util.random_rotation_translation(0.25)
        mvp    = proj_mtx @ mv
        campos = torch.linalg.inv(mv)[:3, 3]

        return mv[None, ...].cuda(), mvp[None, ...].cuda(), campos[None, ...].cuda(), iter_res, self.FLAGS.spp # Add batch dimension

    def __len__(self):
        return 50 if self.validate else (self.FLAGS.iter + 1) * self.FLAGS.batch

    def __getitem__(self, itr):
        # ==============================================================================================
        #  Randomize scene parameters
        # ==============================================================================================

        if self.validate:
            mv, mvp, campos, iter_res, iter_spp = self._rotate_scene(itr)
        else:
            mv, mvp, campos, iter_res, iter_spp = self._random_scene()

        img = render.render_mesh(self.glctx, self.ref_mesh, mvp, campos, self.envlight, iter_res, spp=iter_spp, 
                                num_layers=self.FLAGS.layers, msaa=True, background=None)['shaded']

        return {
            'mv' : mv,
            'mvp' : mvp,
            'campos' : campos,
            'resolution' : iter_res,
            'spp' : iter_spp,
            'img' : img
        }
