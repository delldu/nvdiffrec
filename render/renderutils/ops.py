# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.

import numpy as np
import os
import sys
import torch
import torch.utils.cpp_extension

from .bsdf import *
from .loss import *
import pdb

#----------------------------------------------------------------------------
# C++/Cuda plugin compiler/loader.

_cached_plugin = None
def _get_plugin():
    # ==> Here !!!

    # Return cached plugin if already loaded.
    global _cached_plugin
    if _cached_plugin is not None: # False
        return _cached_plugin

    # Make sure we can find the necessary compiler and libary binaries.
    if os.name == 'nt':
        def find_cl_path():
            import glob
            for edition in ['Enterprise', 'Professional', 'BuildTools', 'Community']:
                paths = sorted(glob.glob(r"C:\Program Files (x86)\Microsoft Visual Studio\*\%s\VC\Tools\MSVC\*\bin\Hostx64\x64" % edition), reverse=True)
                if paths:
                    return paths[0]

        # If cl.exe is not on path, try to find it.
        if os.system("where cl.exe >nul 2>nul") != 0:
            cl_path = find_cl_path()
            if cl_path is None:
                raise RuntimeError("Could not locate a supported Microsoft Visual C++ installation")
            os.environ['PATH'] += ';' + cl_path

    # Compiler options.
    opts = ['-DNVDR_TORCH']

    # Linker options.
    if os.name == 'posix':
        ldflags = ['-lcuda', '-lnvrtc']
    elif os.name == 'nt':
        ldflags = ['cuda.lib', 'advapi32.lib', 'nvrtc.lib']

    # List of sources.
    source_files = [
        'c_src/mesh.cu',
        # 'c_src/loss.cu',
        # 'c_src/bsdf.cu',
        'c_src/normal.cu',
        'c_src/cubemap.cu',
        'c_src/common.cpp',
        'c_src/torch_bindings.cpp'
    ]

    # source_files ====>>>>> 
    # ['c_src/mesh.cu',
    #  'c_src/loss.cu', 
    #  'c_src/bsdf.cu',
    #  'c_src/normal.cu', 
    #  'c_src/cubemap.cu',
    #  'c_src/common.cpp',
    #  'c_src/torch_bindings.cpp']    

    # Some containers set this to contain old architectures that won't compile. We only need the one installed in the machine.
    os.environ['TORCH_CUDA_ARCH_LIST'] = ''

    # Try to detect if a stray lock file is left in cache directory and show a warning. This sometimes happens on Windows if the build is interrupted at just the right moment.
    try:
        lock_fn = os.path.join(torch.utils.cpp_extension._get_build_directory('renderutils_plugin', False), 'lock')
        # lock_fn -- '/home/dell/.cache/torch_extensions/renderutils_plugin/lock'
        if os.path.exists(lock_fn):
            print("Warning: Lock file exists in build directory: '%s'" % lock_fn)
    except:
        pass

    # Compile and load.
    source_paths = [os.path.join(os.path.dirname(__file__), fn) for fn in source_files]
    torch.utils.cpp_extension.load(name='renderutils_plugin', sources=source_paths, extra_cflags=opts,
         extra_cuda_cflags=opts, extra_ldflags=ldflags, with_cuda=True, verbose=True)

    # opts -- ['-DNVDR_TORCH']

    # ldflags --
    #     ['-lcuda', '-lnvrtc', 
    # '-L/home/dell/anaconda3/envs/python39/lib/python3.9/site-packages/torch/lib', 
    #     '-lc10', '-lc10_cuda', '-ltorch_cpu', '-ltorch_cuda', '-ltorch', '-ltorch_python', 
    # '-L/usr/local/cuda-10.2/lib64', '-lcudart']

    # Import, cache, and return the compiled module.
    import renderutils_plugin
    _cached_plugin = renderutils_plugin

    # _cached_plugin -- <module 'renderutils_plugin' 
    #   from '/home/dell/.cache/torch_extensions/renderutils_plugin/renderutils_plugin.so'>

    return _cached_plugin

#----------------------------------------------------------------------------
# Shading normal setup (bump mapping + bent normals)
class shading_normal_function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pos, view_pos, perturbed_nrm, smooth_nrm, smooth_tng, geom_nrm, two_sided_shading, opengl):
        # ==> pdb.set_trace()
        # two_sided_shading = True
        # opengl = True
        ctx.two_sided_shading, ctx.opengl = two_sided_shading, opengl
        out = _get_plugin().shading_normal_forward(pos, view_pos, perturbed_nrm, smooth_nrm, smooth_tng, geom_nrm, two_sided_shading, opengl, False)
        ctx.save_for_backward(pos, view_pos, perturbed_nrm, smooth_nrm, smooth_tng, geom_nrm)
        return out

    @staticmethod
    def backward(ctx, dout):
        # ==> pdb.set_trace()
        pos, view_pos, perturbed_nrm, smooth_nrm, smooth_tng, geom_nrm = ctx.saved_variables
        return _get_plugin().shading_normal_backward(pos, view_pos, perturbed_nrm, smooth_nrm, smooth_tng, geom_nrm, dout, ctx.two_sided_shading, ctx.opengl) + (None, None, None)

def shading_normal(pos, view_pos, perturbed_nrm, smooth_nrm, smooth_tng, geom_nrm, two_sided_shading=True, opengl=True, use_python=False):
    '''Takes care of all corner cases and produces a final normal used for shading:
        - Constructs tangent space
        - Flips normal direction based on geometric normal for two sided Shading
        - Perturbs shading normal by normal map
        - Bends backfacing normals towards the camera to avoid shading artifacts

        All tensors assume a shape of [minibatch_size, height, width, 3] or broadcastable equivalent.

    Args:
        pos: World space g-buffer position.
        view_pos: Camera position in world space (typically using broadcasting).
        perturbed_nrm: Trangent-space normal perturbation from normal map lookup.
        smooth_nrm: Interpolated vertex normals.
        smooth_tng: Interpolated vertex tangents.
        geom_nrm: Geometric (face) normals.
        two_sided_shading: Use one/two sided shading
        opengl: Use OpenGL/DirectX normal map conventions 
        use_python: Use PyTorch implementation (for validation)
    Returns:
        Final shading normal
    '''
    # two_sided_shading = True
    # opengl = True
    # use_python = False

    if perturbed_nrm is None: # True
        perturbed_nrm = torch.tensor([0, 0, 1], dtype=torch.float32, device='cuda', requires_grad=False)[None, None, None, ...]
    
    if use_python:
        out = bsdf_prepare_shading_normal(pos, view_pos, perturbed_nrm, smooth_nrm, smooth_tng, geom_nrm, two_sided_shading, opengl)
    else:
        out = shading_normal_function.apply(pos, view_pos, perturbed_nrm, smooth_nrm, smooth_tng, geom_nrm, two_sided_shading, opengl)
    
    if torch.is_anomaly_enabled(): # False
        assert torch.all(torch.isfinite(out)), "Output of shading_normal contains inf or NaN"
    return out

#----------------------------------------------------------------------------
# cubemap filter with filtering across edges

class cubemap_diffuse_function(torch.autograd.Function):
    # ==> Here
    @staticmethod
    def forward(ctx, cubemap):
        # ==> pdb.set_trace()
        out = _get_plugin().cubemap_diffuse_forward(cubemap)
        ctx.save_for_backward(cubemap)
        return out

    @staticmethod
    def backward(ctx, dout):
        # --> pdb.set_trace()       
        cubemap, = ctx.saved_variables
        cubemap_grad = _get_plugin().cubemap_diffuse_backward(cubemap, dout)
        return cubemap_grad, None

def cubemap_diffuse(cubemap, use_python=False):
    # cubemap.size() -- [6, 16, 16, 3]
    # use_python = False
    if use_python:
        assert False
    else:
        out = cubemap_diffuse_function.apply(cubemap)
    if torch.is_anomaly_enabled(): # False
        assert torch.all(torch.isfinite(out)), "Output of cubemap_diffuse contains inf or NaN"
    return out

class cubemap_specular_function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, cubemap, roughness, costheta_cutoff, bounds):
        # cubemap.size() -- [6, 512, 512, 3]
        # roughness, costheta_cutoff -- (0.08, 0.9997669655912325)
        # bounds.size() -- [6, 512, 512, 24]
        out = _get_plugin().cubemap_specular_forward(cubemap, bounds, roughness, costheta_cutoff)
        ctx.save_for_backward(cubemap, bounds)
        ctx.roughness, ctx.theta_cutoff = roughness, costheta_cutoff
        return out

    @staticmethod
    def backward(ctx, dout):
        #--> pdb.set_trace()
        cubemap, bounds = ctx.saved_variables
        cubemap_grad = _get_plugin().cubemap_specular_backward(cubemap, bounds, dout, ctx.roughness, ctx.theta_cutoff)
        return cubemap_grad, None, None, None

# Compute the bounds of the GGX NDF lobe to retain "cutoff" percent of the energy
def __ndfBounds(res, roughness, cutoff):
    # pdb.set_trace()
    def ndfGGX(alphaSqr, costheta):
        # alphaSqr = 4.096e-05
        # costheta = array([1.0000000e+00, 1.0000000e+00, 1.0000000e+00, ..., 3.1415958e-06,
        #     1.5707979e-06, 6.1232340e-17])
        # costheta.shape -- (1000000,)
        costheta = np.clip(costheta, 0.0, 1.0)
        d = (costheta * alphaSqr - costheta) * costheta + 1.0
        return alphaSqr / (d * d * np.pi)

    # Sample out cutoff angle
    nSamples = 1000000
    costheta = np.cos(np.linspace(0, np.pi/2.0, nSamples))
    D = np.cumsum(ndfGGX(roughness**4, costheta))
    idx = np.argmax(D >= D[..., -1] * cutoff)

    # Brute force compute lookup table with bounds
    bounds = _get_plugin().specular_bounds(res, costheta[idx])

    return costheta[idx], bounds
__ndfBoundsDict = {}

def cubemap_specular(cubemap, roughness, cutoff=0.99, use_python=False):
    # ==> pdb.set_trace()
    # cubemap.size() -- [6, 512, 512, 3]
    # roughness = 0.08
    # cutoff = 0.99
    # use_python = False
    assert cubemap.shape[0] == 6 and cubemap.shape[1] == cubemap.shape[2], "Bad shape for cubemap tensor: %s" % str(cubemap.shape)

    if use_python:
        assert False
    else:
        key = (cubemap.shape[1], roughness, cutoff)
        if key not in __ndfBoundsDict:
            __ndfBoundsDict[key] = __ndfBounds(*key)
        #  __ndfBoundsDict.keys() -- dict_keys([(512, 0.08, 0.99)])
        # __ndfBoundsDict[(512, 0.08, 0.99)][0] -- 0.9997669655912325
        # __ndfBoundsDict[(512, 0.08, 0.99)][1].size() -- [6, 512, 512, 24]
        out = cubemap_specular_function.apply(cubemap, roughness, *__ndfBoundsDict[key])
    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(out)), "Output of cubemap_specular contains inf or NaN"
    return out[..., 0:3] / out[..., 3:]


#----------------------------------------------------------------------------
# Transform points function

class points_transform_function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, points, matrix, isPoints):
        ctx.save_for_backward(points, matrix)
        ctx.isPoints = isPoints
        return _get_plugin().points_transform_forward(points, matrix, isPoints, False)

    @staticmethod
    def backward(ctx, dout):
        points, matrix = ctx.saved_variables
        return (_get_plugin().points_transform_backward(points, matrix, dout, ctx.isPoints),) + (None, None, None)

def points_transform(points, matrix):
    '''Transform points.
    Args:
        points: Tensor containing 3D points with shape [minibatch_size, num_vertices, 3] or [1, num_vertices, 3]
        matrix: A 4x4 transform matrix with shape [minibatch_size, 4, 4]
        use_python: Use PyTorch's torch.matmul (for validation)
    Returns:
        Transformed points in homogeneous 4D with shape [minibatch_size, num_vertices, 4].
    '''
    # points.size() -- [1, 5344, 3]
    # matrix.size() -- [1, 4, 4]
    out = points_transform_function.apply(points, matrix, True)

    if torch.is_anomaly_enabled(): # False
        assert torch.all(torch.isfinite(out)), "Output of points_transform contains inf or NaN"
    return out
