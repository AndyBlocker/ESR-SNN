// st_bif_ss_cuda_kernel.cu
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace {

template <typename scalar_t>
__device__ inline float to_float(scalar_t x) {
  return static_cast<float>(x);
}
template <>
__device__ inline float to_float<c10::Half>(c10::Half x) {
  return __half2float(x);
}

template <typename scalar_t>
__device__ inline scalar_t from_float(float x) {
  return static_cast<scalar_t>(x);
}
template <>
__device__ inline c10::Half from_float<c10::Half>(float x) {
  return __float2half(x);
}

template <typename scalar_t>
__device__ inline float theta_strict(float x) {
  return x > 0.f ? 1.f : 0.f;
}

template <typename scalar_t>
__device__ inline float theta_eq(float x) {
  return x >= 0.f ? 1.f : 0.f;
}

__device__ inline float dtheta_gauss_norm_f32(float x, float vthr) {
  const float sqrt_2pi = 2.506628253f;
  float sigma = fmaxf(vthr / sqrt_2pi, 1e-3f);
  float a = 1.f / (sigma * sqrt_2pi);
  float u = -(x * x) / (2.f * sigma * sigma);
  return a * __expf(u);
}

template <typename scalar_t>
__device__ inline float load_param(const scalar_t* p, bool scalar_mode, int64_t i) {
  return to_float( scalar_mode ? p[0] : p[i] );
}

template <typename scalar_t>
__global__ void st_bifnode_forward_kernel(
    int64_t N,
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ V_prev,
    const scalar_t* __restrict__ T_prev,
    const scalar_t* __restrict__ v_th,
    const scalar_t* __restrict__ T_max,
    const scalar_t* __restrict__ T_min,
    bool vth_scalar, bool Tmax_scalar, bool Tmin_scalar,
    scalar_t* __restrict__ spike,
    scalar_t* __restrict__ V_out,
    scalar_t* __restrict__ T_out,
    scalar_t* __restrict__ H_save // for backward
) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t stride = blockDim.x * gridDim.x;

  for (int64_t i = idx; i < N; i += stride) {
    float xi = to_float(x[i]);
    float Vi = to_float(V_prev[i]);
    float Ti = to_float(T_prev[i]);

    float vth = load_param(v_th, vth_scalar, i);
    float tmax = load_param(T_max, Tmax_scalar, i);
    float tmin = load_param(T_min, Tmin_scalar, i);

    float H = Vi + xi;

    // pos: (H >= v_th) & (T_prev - T_max < 0)
    // neg: (H < 0) & (T_prev - T_min > 0)
    bool cond_pos = (H >= vth) && ((Ti - tmax) < 0.f);
    bool cond_neg = (H < 0.f) && ((Ti - tmin) > 0.f);

    float s = 0.f;
    if (cond_pos) s =  1.f;
    else if (cond_neg) s = -1.f;

    float Vt = H - vth * s;
    float Tt = Ti + s;

    spike[i] = from_float<scalar_t>(s);
    V_out[i] = from_float<scalar_t>(Vt);
    T_out[i] = from_float<scalar_t>(Tt);
    H_save[i] = from_float<scalar_t>(H);
  }
}

template <typename scalar_t>
__global__ void st_bifnode_backward_kernel(
    int64_t N,
    const scalar_t* __restrict__ g_spike,
    const scalar_t* __restrict__ g_V,
    const scalar_t* __restrict__ g_T,
    const scalar_t* __restrict__ H_save,
    const scalar_t* __restrict__ T_prev,
    const scalar_t* __restrict__ v_th,
    const scalar_t* __restrict__ T_max,
    const scalar_t* __restrict__ T_min,
    bool vth_scalar, bool Tmax_scalar, bool Tmin_scalar,
    scalar_t* __restrict__ g_x,
    scalar_t* __restrict__ g_Vprev,
    scalar_t* __restrict__ g_Tprev
) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t stride = blockDim.x * gridDim.x;

  for (int64_t i = idx; i < N; i += stride) {
    float gs = to_float(g_spike[i]);
    float gv = to_float(g_V[i]);
    float gT = to_float(g_T[i]);

    float H  = to_float(H_save[i]);
    float Ti = to_float(T_prev[i]);

    float vth = load_param(v_th, vth_scalar, i);
    float tmax = load_param(T_max, Tmax_scalar, i);
    float tmin = load_param(T_min, Tmin_scalar, i);

    // grad_T_t_to_H_t = dtheta(H - v_th)*theta(T_max - T_prev) + dtheta(-H)*theta(T_prev - T_min)
    float g1 = dtheta_gauss_norm_f32(H - vth, vth);
    float g2 = dtheta_gauss_norm_f32(-H,      vth);
    float th_Tmax = theta_strict<scalar_t>(tmax - Ti);
    float th_Tmin = theta_strict<scalar_t>(Ti   - tmin);

    float grad_Tt_to_H = g1 * th_Tmax + g2 * th_Tmin;

    // grad_Y_t_to_T_prev = -(theta_eq(H - v_th)*dtheta(T_max - T_prev) + theta(-H)*dtheta(T_prev - T_min))
    float th_eq_H = theta_eq<scalar_t>(H - vth);
    float th_negH = theta_strict<scalar_t>(-H);

    float dg_Tmax = dtheta_gauss_norm_f32(tmax - Ti, vth);
    float dg_Tmin = dtheta_gauss_norm_f32(Ti   - tmin, vth);

    float grad_Yt_to_Tprev = -(th_eq_H * dg_Tmax + th_negH * dg_Tmin);

    // tmp = g_spike - v_th * g_V + g_T
    float tmp = gs - vth * gv + gT;

    float gx  = tmp * grad_Tt_to_H + gv;
    float gTp = tmp * grad_Yt_to_Tprev + gT;
    float gVp = gx;

    g_x[i]      = from_float<scalar_t>(gx);
    g_Vprev[i]  = from_float<scalar_t>(gVp);
    g_Tprev[i]  = from_float<scalar_t>(gTp);
  }
}

inline void check_broadcastable(const at::Tensor& base, const at::Tensor& other, const char* name) {
  if (!(other.numel() == 1 || other.sizes() == base.sizes())) {
    TORCH_CHECK(false, name, " must be a scalar or have the same shape as x.");
  }
}

} // namespace

std::vector<at::Tensor> st_bifnode_forward_cuda(
    const at::Tensor& x,
    const at::Tensor& V_prev,
    const at::Tensor& T_prev,
    const at::Tensor& v_th,
    const at::Tensor& T_max,
    const at::Tensor& T_min) {

  TORCH_CHECK(x.is_cuda(), "x must be CUDA");
  TORCH_CHECK(V_prev.is_cuda() && T_prev.is_cuda(), "states must be CUDA");
  TORCH_CHECK(v_th.is_cuda() && T_max.is_cuda() && T_min.is_cuda(), "params must be CUDA");
  TORCH_CHECK(x.numel() == V_prev.numel() && x.numel() == T_prev.numel(), "x/V_prev/T_prev shape mismatch");

  check_broadcastable(x, v_th, "v_th");
  check_broadcastable(x, T_max, "T_max");
  check_broadcastable(x, T_min, "T_min");

  auto x_c = x.contiguous();
  auto V_c = V_prev.contiguous();
  auto T_c = T_prev.contiguous();
  auto v_c = v_th.contiguous();
  auto Tmax_c = T_max.contiguous();
  auto Tmin_c = T_min.contiguous();

  auto spike = at::empty_like(x_c);
  auto V_out = at::empty_like(x_c);
  auto T_out = at::empty_like(x_c);
  auto H_save = at::empty_like(x_c);

  const int64_t N = x_c.numel();
  const int threads = 256;
  const int blocks = std::min<int64_t>(
      (N + threads - 1) / threads,
      (int64_t)at::cuda::getCurrentDeviceProperties()->multiProcessorCount * 8);

  const bool v_scalar  = (v_c.numel() == 1);
  const bool Tmax_scalar = (Tmax_c.numel() == 1);
  const bool Tmin_scalar = (Tmin_c.numel() == 1);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(x_c.scalar_type(), "st_bifnode_forward_cuda", [&] {
    st_bifnode_forward_kernel<scalar_t><<<blocks, threads>>>(
      N,
      x_c.data_ptr<scalar_t>(),
      V_c.data_ptr<scalar_t>(),
      T_c.data_ptr<scalar_t>(),
      v_c.data_ptr<scalar_t>(),
      Tmax_c.data_ptr<scalar_t>(),
      Tmin_c.data_ptr<scalar_t>(),
      v_scalar, Tmax_scalar, Tmin_scalar,
      spike.data_ptr<scalar_t>(),
      V_out.data_ptr<scalar_t>(),
      T_out.data_ptr<scalar_t>(),
      H_save.data_ptr<scalar_t>()
    );
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return {spike, V_out, T_out, H_save};
}

std::vector<at::Tensor> st_bifnode_backward_cuda(
    const at::Tensor& g_spike,
    const at::Tensor& g_V,
    const at::Tensor& g_T,
    const at::Tensor& H_save,
    const at::Tensor& T_prev,
    const at::Tensor& v_th,
    const at::Tensor& T_max,
    const at::Tensor& T_min) {

  TORCH_CHECK(g_spike.is_cuda() && g_V.is_cuda() && g_T.is_cuda(), "grads must be CUDA");
  TORCH_CHECK(H_save.is_cuda() && T_prev.is_cuda(), "saved must be CUDA");
  TORCH_CHECK(v_th.is_cuda() && T_max.is_cuda() && T_min.is_cuda(), "params must be CUDA");
  TORCH_CHECK(g_spike.numel() == H_save.numel(), "size mismatch");

  check_broadcastable(H_save, v_th, "v_th");
  check_broadcastable(H_save, T_max, "T_max");
  check_broadcastable(H_save, T_min, "T_min");

  auto gs = g_spike.contiguous();
  auto gv = g_V.contiguous();
  auto gT = g_T.contiguous();
  auto Hc = H_save.contiguous();
  auto Tp = T_prev.contiguous();
  auto v_c = v_th.contiguous();
  auto Tmax_c = T_max.contiguous();
  auto Tmin_c = T_min.contiguous();

  auto gx  = at::empty_like(gs);
  auto gVp = at::empty_like(gs);
  auto gTp = at::empty_like(gs);

  const int64_t N = gs.numel();
  const int threads = 256;
  const int blocks = std::min<int64_t>(
      (N + threads - 1) / threads,
      (int64_t)at::cuda::getCurrentDeviceProperties()->multiProcessorCount * 8);

  const bool v_scalar  = (v_c.numel() == 1);
  const bool Tmax_scalar = (Tmax_c.numel() == 1);
  const bool Tmin_scalar = (Tmin_c.numel() == 1);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(gs.scalar_type(), "st_bifnode_backward_cuda", [&] {
    st_bifnode_backward_kernel<scalar_t><<<blocks, threads>>>(
      N,
      gs.data_ptr<scalar_t>(),
      gv.data_ptr<scalar_t>(),
      gT.data_ptr<scalar_t>(),
      Hc.data_ptr<scalar_t>(),
      Tp.data_ptr<scalar_t>(),
      v_c.data_ptr<scalar_t>(),
      Tmax_c.data_ptr<scalar_t>(),
      Tmin_c.data_ptr<scalar_t>(),
      v_scalar, Tmax_scalar, Tmin_scalar,
      gx.data_ptr<scalar_t>(),
      gVp.data_ptr<scalar_t>(),
      gTp.data_ptr<scalar_t>()
    );
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return {gx, gVp, gTp};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("st_bifnode_forward",  &st_bifnode_forward_cuda,  "ST_BIFNodeATGF_SS forward (CUDA)");
  m.def("st_bifnode_backward", &st_bifnode_backward_cuda, "ST_BIFNodeATGF_SS backward (CUDA)");
}
