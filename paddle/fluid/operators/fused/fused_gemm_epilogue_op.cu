/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
Copyright (c) 2022 NVIDIA Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <algorithm>
#include <mutex>
#include <unordered_map>
#include "paddle/fluid/platform/float16.h"

#include "gflags/gflags.h"
#include "glog/logging.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/backends/gpu/rocm/rocm_helper.h"
#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/scope_guard.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/fc_functor.h"
#include "paddle/phi/kernels/gpu/reduce.h"
#include "paddle/phi/kernels/primitive/functor_primitives.h"
#include "paddle/utils/optional.h"

namespace paddle {
namespace operators {

template <typename T>
struct FcTypeTraits;

template <>
struct FcTypeTraits<float> {
  typedef float4 Type;
};

template <>
struct FcTypeTraits<double> {
  typedef double4 Type;
};

struct float16_4 {
  float16 x, y, z, w;
};

template <>
struct FcTypeTraits<float16> {
  typedef float16_4 Type;
};

template <typename T, bool DoRelu>
__global__ void bias_relu_v4(const int num, const T* bias, T* data, int K) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < num) {
    int bias_idx = tid % K;
    const T bias_ptr = bias[bias_idx];
    const T in_ptr = data[tid];
    T packed_val;
    packed_val.x = in_ptr.x + bias_ptr.x;
    packed_val.y = in_ptr.y + bias_ptr.y;
    packed_val.z = in_ptr.z + bias_ptr.z;
    packed_val.w = in_ptr.w + bias_ptr.w;
    if (DoRelu) {
      packed_val.x = fmaxf(0.f, packed_val.x);
      packed_val.y = fmaxf(0.f, packed_val.y);
      packed_val.z = fmaxf(0.f, packed_val.z);
      packed_val.w = fmaxf(0.f, packed_val.w);
    }
    data[tid] = packed_val;
  }
}

template <typename T, bool DoRelu, int BlockDim>
__global__ void InplaceAddReluKernel(const int N, const T* bias, T* data) {
  int offset = blockIdx.x * N;

  for (int i = threadIdx.x; i < N; i += BlockDim) {
    T temp;
#if defined(__HIPCC__) || __CUDA_ARCH__ >= 350
    temp = __ldg(data + offset + i) + __ldg(bias + i);
#else
    temp = data[offset + i] + bias[i];
#endif
    if (DoRelu) {
      data[offset + i] = static_cast<int>(temp > 0) * temp;
    } else {
      data[offset + i] = temp;
    }
  }
}

template <bool DoRelu, int BlockDim>
__global__ void InplaceAddReluKernel(const int N,
                                     const float16* bias,
                                     float16* data) {
  int offset = blockIdx.x * N;
  for (int i = threadIdx.x; i < N; i += BlockDim) {
    float16 temp;
    temp = data[offset + i] + bias[i];
    if (DoRelu) {
      data[offset + i] = fmaxf(0.f, temp);
    } else {
      data[offset + i] = temp;
    }
  }
}

template <typename T>
void AddReluKernel(
    gpuStream_t stream, const int M, const int N, T* Y, const T* B, bool relu) {
  if (N % 4 == 0) {
    const int threads = 256;
    const int num = M * N / 4;
    const int blocks = (num + threads - 1) / threads;
    typedef typename FcTypeTraits<T>::Type trans_type;
    auto* bias_ptr_v4 = reinterpret_cast<const trans_type*>(B);
    auto* data_ptr_v4 = reinterpret_cast<trans_type*>(Y);
    if (relu) {
      bias_relu_v4<trans_type, true><<<blocks, threads, 0, stream>>>(
          num, bias_ptr_v4, data_ptr_v4, N / 4);
    } else {
      bias_relu_v4<trans_type, false><<<blocks, threads, 0, stream>>>(
          num, bias_ptr_v4, data_ptr_v4, N / 4);
    }
  } else {
    const int threads = 256;
    const int blocks = M;

    if (relu) {
      InplaceAddReluKernel<T, true, threads>
          <<<blocks, threads, 0, stream>>>(N, B, Y);
    } else {
      InplaceAddReluKernel<T, false, threads>
          <<<blocks, threads, 0, stream>>>(N, B, Y);
    }
  }
}

template <>
void AddReluKernel(gpuStream_t stream,
                   const int M,
                   const int N,
                   float16* Y,
                   const float16* B,
                   bool relu) {
  if (N % 4 == 0) {
    const int threads = 256;
    const int num = M * N / 4;
    const int blocks = (num + threads - 1) / threads;
    typedef typename FcTypeTraits<float16>::Type trans_type;
    auto* bias_ptr_v4 = reinterpret_cast<const trans_type*>(B);
    auto* data_ptr_v4 = reinterpret_cast<trans_type*>(Y);
    if (relu) {
      bias_relu_v4<trans_type, true><<<blocks, threads, 0, stream>>>(
          num, bias_ptr_v4, data_ptr_v4, N / 4);
    } else {
      bias_relu_v4<trans_type, false><<<blocks, threads, 0, stream>>>(
          num, bias_ptr_v4, data_ptr_v4, N / 4);
    }
  } else {
    const int threads = 256;
    const int blocks = M;

    if (relu) {
      InplaceAddReluKernel<true, threads>
          <<<blocks, threads, 0, stream>>>(N, B, Y);
    } else {
      InplaceAddReluKernel<false, threads>
          <<<blocks, threads, 0, stream>>>(N, B, Y);
    }
  }
}

template <typename T, typename DeviceContext>
class FusedGemmEpilogueKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto& dev_ctx = ctx.template device_context<phi::GPUContext>();

    const phi::DenseTensor* x = ctx.Input<phi::DenseTensor>("X");
    const phi::DenseTensor* y = ctx.Input<phi::DenseTensor>("Y");
    const phi::DenseTensor* bias = ctx.Input<phi::DenseTensor>("Bias");
    phi::DenseTensor* out = ctx.Output<phi::DenseTensor>("Out");
    phi::DenseTensor* reserve_space =
        ctx.Output<phi::DenseTensor>("ReserveSpace");

    bool trans_x = ctx.Attr<bool>("trans_x");
    bool trans_y = ctx.Attr<bool>("trans_y");

    std::string activation = ctx.Attr<std::string>("activation");
    dev_ctx.Alloc<T>(out, out->numel() * sizeof(T));
    // (M * K) * (K * N)
    auto x_mat_dims =
        phi::flatten_to_2d(x->dims(), trans_x ? 1 : x->dims().size() - 1);
    int64_t M = trans_x ? x_mat_dims[1] : x_mat_dims[0];
    int64_t K = trans_y ? y->dims()[1] : y->dims()[0];
    int64_t N = trans_y ? y->dims()[0] : y->dims()[1];

    void* reserve_data = reserve_space ? reserve_space->data() : nullptr;
    VLOG(6) << "running FusedGemmEpilogueKernel on DCU";
    auto blas = phi::funcs::GetBlas<DeviceContext, T>(dev_ctx);
    blas.GEMM(CblasNoTrans,
              CblasNoTrans,
              M,
              N,
              K,
              static_cast<T>(1.0),
              x->data<T>(),
              y->data<T>(),
              static_cast<T>(0.0),
              out->data<T>());

    auto B = bias->data<T>();
    AddReluKernel(dev_ctx.stream(),
                  static_cast<int>(M),
                  static_cast<int>(N),
                  out->data<T>(),
                  B,
                  false);
  }
};

template <typename T, typename DeviceContext>
class FusedGemmEpilogueGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto& dev_ctx = ctx.template device_context<phi::GPUContext>();

    const phi::DenseTensor* dout = ctx.Input<phi::DenseTensor>("DOut");
    const phi::DenseTensor* x = ctx.Input<phi::DenseTensor>("X");
    const phi::DenseTensor* y = ctx.Input<phi::DenseTensor>("Y");
    const phi::DenseTensor* reserve_space =
        ctx.Input<phi::DenseTensor>("ReserveSpace");
    phi::DenseTensor* dx = ctx.Output<phi::DenseTensor>("DX");
    phi::DenseTensor* dy = ctx.Output<phi::DenseTensor>("DY");
    phi::DenseTensor* dbias = ctx.Output<phi::DenseTensor>("DBias");

    std::string activation_grad = ctx.Attr<std::string>("activation_grad");
    bool trans_x = ctx.Attr<bool>("trans_x");
    bool trans_y = ctx.Attr<bool>("trans_y");

    // (M * K) * (K * N)
    auto x_mat_dims =
        phi::flatten_to_2d(x->dims(), trans_x ? 1 : x->dims().size() - 1);
    int64_t M = trans_x ? x_mat_dims[1] : x_mat_dims[0];
    int64_t K = trans_y ? y->dims()[1] : y->dims()[0];
    int64_t N = trans_y ? y->dims()[0] : y->dims()[1];

    VLOG(6) << "Running FusedGemmEpilogueKernel backward on DCU";

    auto blas = phi::funcs::GetBlas<DeviceContext, T>(dev_ctx);
    // fc grad
    if (dx) {
      // dx = matmul(dz, y, None, False, True)
      dev_ctx.Alloc<T>(dx);
      blas.GEMM(CblasNoTrans,
                CblasTrans,
                M,
                K,
                N,
                static_cast<T>(1.0),
                dout->data<T>(),
                y->data<T>(),
                static_cast<T>(0.0),
                dx->data<T>());
    }
    if (dy) {
      dev_ctx.Alloc<T>(dy);
      // dy = matmul(x, dz, None, True, False)
      blas.GEMM(CblasTrans,
                CblasNoTrans,
                K,
                N,
                M,
                static_cast<T>(1.0),
                x->data<T>(),
                dout->data<T>(),
                static_cast<T>(0.0),
                dy->data<T>());
    }
    // bas grad
    if (dbias) {
      dev_ctx.Alloc<T>(dbias);
      //  dbias = np.sum(dz, axis=0, keepdims=False)
      const std::vector<int64_t> reduce_dims{0};
      phi::Reduce<T, phi::kps::AddFunctor, phi::kps::IdentityFunctor>(
          dev_ctx, *dout, false, reduce_dims, false, dout->dtype(), dbias);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
PD_REGISTER_STRUCT_KERNEL(fused_gemm_epilogue,
                          GPU,
                          ALL_LAYOUT,
                          ops::FusedGemmEpilogueKernel,
                          float,
                          double,
                          plat::float16) {}

PD_REGISTER_STRUCT_KERNEL(fused_gemm_epilogue_grad,
                          GPU,
                          ALL_LAYOUT,
                          ops::FusedGemmEpilogueGradKernel,
                          float,
                          double,
                          plat::float16) {}
