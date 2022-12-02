/*
 * Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related 
 * documentation and any modifications thereto. Any use, reproduction, 
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or 
 * its affiliates is strictly prohibited.
 */

#ifdef _MSC_VER 
#pragma warning(push, 0)
#include <torch/extension.h>
#pragma warning(pop)
#else
#include <torch/extension.h>
#endif

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAUtils.h>
#include <algorithm>
#include <string>

#define NVDR_CHECK_CUDA_ERROR(CUDA_CALL) { cudaError_t err = CUDA_CALL; AT_CUDA_CHECK(cudaGetLastError()); }
#define NVDR_CHECK_GL_ERROR(GL_CALL) { GL_CALL; GLenum err = glGetError(); TORCH_CHECK(err == GL_NO_ERROR, "OpenGL error: ", getGLErrorString(err), "[", #GL_CALL, ";]"); }
#define CHECK_TENSOR(X, DIMS, CHANNELS) \
    TORCH_CHECK(X.is_cuda(), #X " must be a cuda tensor") \
    TORCH_CHECK(X.scalar_type() == torch::kFloat || X.scalar_type() == torch::kBFloat16, #X " must be fp32 or bf16") \
    TORCH_CHECK(X.dim() == DIMS, #X " must have " #DIMS " dimensions") \
    TORCH_CHECK(X.size(DIMS - 1) == CHANNELS, #X " must have " #CHANNELS " channels")

#include "common.h"
#include "loss.h"
#include "normal.h"
#include "cubemap.h"
#include "bsdf.h"
#include "mesh.h"

#define BLOCK_X 8
#define BLOCK_Y 8

//------------------------------------------------------------------------
// mesh.cu

void PointsTransformFowardKernel(PointsTransformKernelParams p);
void PointsTransformBackwardKernel(PointsTransformKernelParams p);

//------------------------------------------------------------------------
// normal.cu

void ShadingNormalForwardKernel(ShadingNormalKernelParams p);
void ShadingNormalBackwardKernel(ShadingNormalKernelParams p);

//------------------------------------------------------------------------
// cubemap.cu

void CubemapDiffuseForwardKernel(CubemapDiffuseKernelParams p);
void CubemapDiffuseBackwardKernel(CubemapDiffuseKernelParams p);
void SpecularBoundsKernel(SpecularBoundsKernelParams p);
void CubemapSpecularForwardKernel(CubemapSpecularKernelParams p);
void CubemapSpecularBackwardKernel(CubemapSpecularKernelParams p);

//------------------------------------------------------------------------
// Tensor helpers

void update_grid(dim3 &gridSize, torch::Tensor x)
{
    gridSize.x = std::max(gridSize.x, (uint32_t)x.size(2));
    gridSize.y = std::max(gridSize.y, (uint32_t)x.size(1));
    gridSize.z = std::max(gridSize.z, (uint32_t)x.size(0));
}

template<typename... Ts>
void update_grid(dim3& gridSize, torch::Tensor x, Ts&&... vs)
{
    gridSize.x = std::max(gridSize.x, (uint32_t)x.size(2));
    gridSize.y = std::max(gridSize.y, (uint32_t)x.size(1));
    gridSize.z = std::max(gridSize.z, (uint32_t)x.size(0));
    update_grid(gridSize, std::forward<Ts>(vs)...);
}

Tensor make_cuda_tensor(torch::Tensor val)
{
    Tensor res;
    for (int i = 0; i < val.dim(); ++i)
    {
        res.dims[i] = val.size(i);
        res.strides[i] = val.stride(i);
    }
    res.fp16 = val.scalar_type() == torch::kBFloat16;
    res.val = res.fp16 ? (void*)val.data_ptr<torch::BFloat16>() : (void*)val.data_ptr<float>();
    res.d_val = nullptr;
    return res;
}

Tensor make_cuda_tensor(torch::Tensor val, dim3 outDims, torch::Tensor* grad = nullptr)
{
    Tensor res;
    for (int i = 0; i < val.dim(); ++i)
    {
        res.dims[i] = val.size(i);
        res.strides[i] = val.stride(i);
    }
    if (val.dim() == 4)
        res._dims[0] = outDims.z, res._dims[1] = outDims.y, res._dims[2] = outDims.x, res._dims[3] = val.size(3);
    else
        res._dims[0] = outDims.z, res._dims[1] = outDims.x, res._dims[2] = val.size(2), res._dims[3] = 1; // Add a trailing one for indexing math to work out

    res.fp16 = val.scalar_type() == torch::kBFloat16;
    res.val = res.fp16 ? (void*)val.data_ptr<torch::BFloat16>() : (void*)val.data_ptr<float>();
    res.d_val = nullptr;
    if (grad != nullptr)
    {
        if (val.dim() == 4)
            *grad = torch::empty({ outDims.z, outDims.y, outDims.x, val.size(3) }, torch::TensorOptions().dtype(res.fp16 ? torch::kBFloat16 : torch::kFloat32).device(torch::kCUDA));
        else // 3
            *grad = torch::empty({ outDims.z, outDims.x, val.size(2) }, torch::TensorOptions().dtype(res.fp16 ? torch::kBFloat16 : torch::kFloat32).device(torch::kCUDA));

        res.d_val = res.fp16 ? (void*)grad->data_ptr<torch::BFloat16>() : (void*)grad->data_ptr<float>();
    }
    return res;
}

//------------------------------------------------------------------------
// shading_normal
torch::Tensor shading_normal_forward(torch::Tensor pos, torch::Tensor view_pos, 
    torch::Tensor perturbed_nrm, torch::Tensor smooth_nrm, torch::Tensor smooth_tng, 
    torch::Tensor geom_nrm, bool two_sided_shading, bool opengl, bool fp16)
{
    CHECK_TENSOR(pos, 4, 3);
    CHECK_TENSOR(view_pos, 4, 3);
    CHECK_TENSOR(perturbed_nrm, 4, 3);
    CHECK_TENSOR(smooth_nrm, 4, 3);
    CHECK_TENSOR(smooth_tng, 4, 3);
    CHECK_TENSOR(geom_nrm, 4, 3);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Extract input parameters.
    ShadingNormalKernelParams p;
    p.two_sided_shading = two_sided_shading;
    p.opengl = opengl;
    p.out.fp16 = fp16;
    update_grid(p.gridSize, pos, view_pos, perturbed_nrm, smooth_nrm, smooth_tng, geom_nrm);

    // Allocate output tensors.
    torch::TensorOptions opts = torch::TensorOptions().dtype(p.out.fp16 ? torch::kBFloat16 : torch::kFloat32).device(torch::kCUDA);
    torch::Tensor out = torch::empty({ p.gridSize.z, p.gridSize.y, p.gridSize.x, 3 }, opts);

    // Choose launch parameters.
    dim3 blockSize = getLaunchBlockSize(BLOCK_X, BLOCK_Y, p.gridSize);
    dim3 gridSize = getLaunchGridSize(blockSize, p.gridSize);

    // Setup tensors
    p.pos = make_cuda_tensor(pos, p.gridSize);
    p.view_pos = make_cuda_tensor(view_pos, p.gridSize);
    p.perturbed_nrm = make_cuda_tensor(perturbed_nrm, p.gridSize);
    p.smooth_nrm = make_cuda_tensor(smooth_nrm, p.gridSize);
    p.smooth_tng = make_cuda_tensor(smooth_tng, p.gridSize);
    p.geom_nrm = make_cuda_tensor(geom_nrm, p.gridSize);
    p.out = make_cuda_tensor(out, p.gridSize);

    // Launch CUDA kernel.
    void* args[] = { &p };
    NVDR_CHECK_CUDA_ERROR(cudaLaunchKernel((const void*)ShadingNormalForwardKernel, gridSize, blockSize, args, 0, stream));

    return out;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> 
shading_normal_backward(torch::Tensor pos, torch::Tensor view_pos, torch::Tensor perturbed_nrm, 
    torch::Tensor smooth_nrm, torch::Tensor smooth_tng, torch::Tensor geom_nrm, torch::Tensor grad, 
    bool two_sided_shading, bool opengl)
{
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Extract input parameters.
    ShadingNormalKernelParams p;
    p.two_sided_shading = two_sided_shading;
    p.opengl = opengl;
    update_grid(p.gridSize, pos, view_pos, perturbed_nrm, smooth_nrm, smooth_tng, geom_nrm);

    // Choose launch parameters.
    dim3 blockSize = getLaunchBlockSize(BLOCK_X, BLOCK_Y, p.gridSize);
    dim3 gridSize = getLaunchGridSize(blockSize, p.gridSize);

    // Setup tensors
    torch::Tensor pos_grad, view_pos_grad, perturbed_nrm_grad, smooth_nrm_grad, smooth_tng_grad, geom_nrm_grad;
    p.pos = make_cuda_tensor(pos, p.gridSize, &pos_grad);
    p.view_pos = make_cuda_tensor(view_pos, p.gridSize, &view_pos_grad);
    p.perturbed_nrm = make_cuda_tensor(perturbed_nrm, p.gridSize, &perturbed_nrm_grad);
    p.smooth_nrm = make_cuda_tensor(smooth_nrm, p.gridSize, &smooth_nrm_grad);
    p.smooth_tng = make_cuda_tensor(smooth_tng, p.gridSize, &smooth_tng_grad);
    p.geom_nrm = make_cuda_tensor(geom_nrm, p.gridSize, &geom_nrm_grad);
    p.out = make_cuda_tensor(grad, p.gridSize);

    // Launch CUDA kernel.
    void* args[] = { &p };
    NVDR_CHECK_CUDA_ERROR(cudaLaunchKernel((const void*)ShadingNormalBackwardKernel, gridSize, blockSize, args, 0, stream));

    return std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>(pos_grad, view_pos_grad, perturbed_nrm_grad, smooth_nrm_grad, smooth_tng_grad, geom_nrm_grad);
}

//------------------------------------------------------------------------
// filter_cubemap
torch::Tensor cubemap_diffuse_forward(torch::Tensor cubemap)
{
    CHECK_TENSOR(cubemap, 4, 3);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Extract input parameters.
    CubemapDiffuseKernelParams p;
    update_grid(p.gridSize, cubemap);

    // Allocate output tensors.
    torch::TensorOptions opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::Tensor out = torch::empty({ p.gridSize.z, p.gridSize.y, p.gridSize.x, 3 }, opts);

    // Choose launch parameters.
    dim3 blockSize = getLaunchBlockSize(BLOCK_X, BLOCK_Y, p.gridSize);
    dim3 gridSize = getLaunchGridSize(blockSize, p.gridSize);

    // Setup tensors
    p.cubemap = make_cuda_tensor(cubemap, p.gridSize);
    p.out = make_cuda_tensor(out, p.gridSize);

    // Launch CUDA kernel.
    void* args[] = { &p };
    NVDR_CHECK_CUDA_ERROR(cudaLaunchKernel((const void*)CubemapDiffuseForwardKernel, gridSize, blockSize, args, 0, stream));

    return out;
}

torch::Tensor cubemap_diffuse_backward(torch::Tensor cubemap, torch::Tensor grad)
{
    CHECK_TENSOR(cubemap, 4, 3);
    CHECK_TENSOR(grad, 4, 3);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Extract input parameters.
    CubemapDiffuseKernelParams p;
    update_grid(p.gridSize, cubemap);

    // Choose launch parameters.
    dim3 blockSize = getLaunchBlockSize(BLOCK_X, BLOCK_Y, p.gridSize);
    dim3 gridSize = getLaunchGridSize(blockSize, p.gridSize);

    // Setup tensors
    torch::Tensor cubemap_grad;
    p.cubemap = make_cuda_tensor(cubemap, p.gridSize);
    p.out = make_cuda_tensor(grad, p.gridSize);

    cubemap_grad = torch::zeros({ p.gridSize.z, p.gridSize.y, p.gridSize.x, cubemap.size(3) }, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    p.cubemap.d_val = (void*)cubemap_grad.data_ptr<float>();

    // Launch CUDA kernel.
    void* args[] = { &p };
    NVDR_CHECK_CUDA_ERROR(cudaLaunchKernel((const void*)CubemapDiffuseBackwardKernel, gridSize, blockSize, args, 0, stream));

    return cubemap_grad;
}

torch::Tensor specular_bounds(int resolution, float costheta_cutoff)
{
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Extract input parameters.
    SpecularBoundsKernelParams p;
    p.costheta_cutoff = costheta_cutoff;
    p.gridSize = dim3(resolution, resolution, 6);

    // Allocate output tensors.
    torch::TensorOptions opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::Tensor out = torch::zeros({ p.gridSize.z, p.gridSize.y, p.gridSize.x, 6*4 }, opts);

    // Choose launch parameters.
    dim3 blockSize = getLaunchBlockSize(BLOCK_X, BLOCK_Y, p.gridSize);
    dim3 gridSize = getLaunchGridSize(blockSize, p.gridSize);

    // Setup tensors
    p.out = make_cuda_tensor(out, p.gridSize);

    // Launch CUDA kernel.
    void* args[] = { &p };
    NVDR_CHECK_CUDA_ERROR(cudaLaunchKernel((const void*)SpecularBoundsKernel, gridSize, blockSize, args, 0, stream));

    return out;
}

torch::Tensor cubemap_specular_forward(torch::Tensor cubemap, torch::Tensor bounds, float roughness, float costheta_cutoff)
{
    CHECK_TENSOR(cubemap, 4, 3);
    CHECK_TENSOR(bounds, 4, 6*4);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Extract input parameters.
    CubemapSpecularKernelParams p;
    p.roughness = roughness;
    p.costheta_cutoff = costheta_cutoff;
    update_grid(p.gridSize, cubemap);

    // Allocate output tensors.
    torch::TensorOptions opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::Tensor out = torch::empty({ p.gridSize.z, p.gridSize.y, p.gridSize.x, 4 }, opts);

    // Choose launch parameters.
    dim3 blockSize = getLaunchBlockSize(BLOCK_X, BLOCK_Y, p.gridSize);
    dim3 gridSize = getLaunchGridSize(blockSize, p.gridSize);

    // Setup tensors
    p.cubemap = make_cuda_tensor(cubemap, p.gridSize);
    p.bounds = make_cuda_tensor(bounds, p.gridSize);
    p.out = make_cuda_tensor(out, p.gridSize);

    // Launch CUDA kernel.
    void* args[] = { &p };
    NVDR_CHECK_CUDA_ERROR(cudaLaunchKernel((const void*)CubemapSpecularForwardKernel, gridSize, blockSize, args, 0, stream));

    return out;
}

torch::Tensor cubemap_specular_backward(torch::Tensor cubemap, torch::Tensor bounds, torch::Tensor grad, float roughness, float costheta_cutoff)
{
    CHECK_TENSOR(cubemap, 4, 3);
    CHECK_TENSOR(bounds, 4, 6*4);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Extract input parameters.
    CubemapSpecularKernelParams p;
    p.roughness = roughness;
    p.costheta_cutoff = costheta_cutoff;
    update_grid(p.gridSize, cubemap);

    // Choose launch parameters.
    dim3 blockSize = getLaunchBlockSize(BLOCK_X, BLOCK_Y, p.gridSize);
    dim3 gridSize = getLaunchGridSize(blockSize, p.gridSize);

    // Setup tensors
    torch::Tensor cubemap_grad;
    p.cubemap = make_cuda_tensor(cubemap, p.gridSize);
    p.bounds = make_cuda_tensor(bounds, p.gridSize);
    p.out = make_cuda_tensor(grad, p.gridSize);

    cubemap_grad = torch::zeros({ p.gridSize.z, p.gridSize.y, p.gridSize.x, cubemap.size(3) }, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    p.cubemap.d_val = (void*)cubemap_grad.data_ptr<float>();

    // Launch CUDA kernel.
    void* args[] = { &p };
    NVDR_CHECK_CUDA_ERROR(cudaLaunchKernel((const void*)CubemapSpecularBackwardKernel, gridSize, blockSize, args, 0, stream));

    return cubemap_grad;
}

//------------------------------------------------------------------------
// transform function

torch::Tensor points_transform_forward(torch::Tensor points, torch::Tensor matrix, bool isPoints, bool fp16)
{
    CHECK_TENSOR(points, 3, 3);
    CHECK_TENSOR(matrix, 3, 4);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Extract input parameters.
    PointsTransformKernelParams p;
    p.out.fp16 = fp16;
    p.isPoints = isPoints;
    p.gridSize.x = points.size(1);
    p.gridSize.y = 1;
    p.gridSize.z = std::max(matrix.size(0), points.size(0));

    // Choose launch parameters.
    dim3 blockSize(BLOCK_X * BLOCK_Y, 1, 1);
    dim3 warpSize = getWarpSize(blockSize);
    dim3 gridSize = getLaunchGridSize(blockSize, p.gridSize);

    // Allocate output tensors.
    torch::TensorOptions opts = torch::TensorOptions().dtype(p.out.fp16 ? torch::kBFloat16 : torch::kFloat32).device(torch::kCUDA);
    torch::Tensor out = isPoints ? torch::empty({ matrix.size(0), points.size(1), 4 }, opts) : torch::empty({ matrix.size(0), points.size(1), 3 }, opts);

    p.points = make_cuda_tensor(points, p.gridSize);
    p.matrix = make_cuda_tensor(matrix, p.gridSize);
    p.out = make_cuda_tensor(out, p.gridSize);

    // Launch CUDA kernel.
    void* args[] = { &p };
    NVDR_CHECK_CUDA_ERROR(cudaLaunchKernel((const void*)PointsTransformFowardKernel, gridSize, blockSize, args, 0, stream));

    return out;
}

torch::Tensor points_transform_backward(torch::Tensor points, torch::Tensor matrix, torch::Tensor grad, bool isPoints)
{
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Extract input parameters.
    PointsTransformKernelParams p;
    p.isPoints = isPoints;
    p.gridSize.x = points.size(1);
    p.gridSize.y = 1;
    p.gridSize.z = std::max(matrix.size(0), points.size(0));

    // Choose launch parameters.
    dim3 blockSize(BLOCK_X * BLOCK_Y, 1, 1);
    dim3 warpSize = getWarpSize(blockSize);
    dim3 gridSize = getLaunchGridSize(blockSize, p.gridSize);

    torch::Tensor points_grad;
    p.points = make_cuda_tensor(points, p.gridSize, &points_grad);
    p.matrix = make_cuda_tensor(matrix, p.gridSize);
    p.out = make_cuda_tensor(grad, p.gridSize);

    // Launch CUDA kernel.
    void* args[] = { &p };
    NVDR_CHECK_CUDA_ERROR(cudaLaunchKernel((const void*)PointsTransformBackwardKernel, gridSize, blockSize, args, 0, stream));

    return points_grad;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("shading_normal_forward", &shading_normal_forward, "shading_normal_forward");
    m.def("shading_normal_backward", &shading_normal_backward, "shading_normal_backward");
    m.def("cubemap_diffuse_forward", &cubemap_diffuse_forward, "cubemap_diffuse_forward");
    m.def("cubemap_diffuse_backward", &cubemap_diffuse_backward, "cubemap_diffuse_backward");

    m.def("specular_bounds", &specular_bounds, "specular_bounds");
    m.def("cubemap_specular_forward", &cubemap_specular_forward, "cubemap_specular_forward");
    m.def("cubemap_specular_backward", &cubemap_specular_backward, "cubemap_specular_backward");

    m.def("points_transform_forward", &points_transform_forward, "points_transform_forward");
    m.def("points_transform_backward", &points_transform_backward, "points_transform_backward");
}
