// cuda_snn_kernels.cu
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>
#include <vector>
#include <limits>

#define CUDA_CHECK(cond)  \
  do { cudaError_t _e = (cond); if (_e != cudaSuccess) { \
    AT_ERROR("CUDA error ", static_cast<int>(_e), " : ", cudaGetErrorString(_e)); } } while (0)

template <typename T>
__device__ inline T one();
template <> __device__ inline float  one<float>()  { return 1.0f; }
template <> __device__ inline double one<double>() { return 1.0;  }

template <typename T>
__device__ inline T zero();
template <> __device__ inline float  zero<float>()  { return 0.0f; }
template <> __device__ inline double zero<double>() { return 0.0;  }

__device__ inline float  theta_fp32(float x)   { return x > 0.f ? 1.f : 0.f; }
__device__ inline float  thetaeq_fp32(float x) { return x >= 0.f ? 1.f : 0.f; }
__device__ inline double theta_fp64(double x)  { return x > 0.0 ? 1.0 : 0.0; }
__device__ inline double thetaeq_fp64(double x){ return x >= 0.0 ? 1.0 : 0.0; }

__device__ inline float dtheta_gauss_f32(float x, float Vthr, float S, float Smin, float Smax) {
  float sigma = 0.405f * Vthr;
  float a = 1.0f / fmaxf(Vthr, 1e-12f);
  float u = -(x * x) / (2.0f * sigma * sigma);
  return a * __expf(u);
}
__device__ inline double dtheta_gauss_f64(double x, double Vthr, double S, double Smin, double Smax) {
  double sigma = 0.405 * Vthr;
  double a = 1.0 / fmax(Vthr, 1e-18);
  double u = -(x * x) / (2.0 * sigma * sigma);
  return a * exp(u);
}

// ===================== Forward kernels =====================

__global__ void forward_kernel_fp32(
    const float* __restrict__ x_seq,
    const float* __restrict__ v_th,
    const float* __restrict__ T_max,
    const float* __restrict__ T_min,
    const float* __restrict__ prefire,
    float* __restrict__ spike_seq_out, // [T+1, B*F]
    float* __restrict__ v_out,         // [B*F]
    float* __restrict__ T_seq_out,     // [T+1, B*F]
    float* __restrict__ H_seq_out,     // [T+1, B*F]
    int64_t time_steps, int64_t spatial)
{
    const int64_t stride = blockDim.x * gridDim.x;
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < spatial; idx += stride)
    {
        float v = (0.5f + prefire[0]) * v_th[0];
        float T = 0.0f;
        spike_seq_out[idx] = 0.0f;
        T_seq_out[idx]     = 0.0f;
        H_seq_out[idx]     = v;

        const float vthr = v_th[0];
        const float tmax = T_max[0];
        const float tmin = T_min[0];
        const float pf_delta = (tmax > 0.f) ? (prefire[0] * vthr / tmax) : 0.f;
        const int   max_pf_steps = (int)fminf(fmaxf(tmax, 0.f), (float)time_steps);

        for (int64_t t = 0; t < time_steps; ++t) {
            const int64_t cur  = t * spatial + idx;
            const int64_t next = (t + 1) * spatial + idx;

            v += x_seq[cur];
            H_seq_out[next] = v;

            float spike_pos = (v >= vthr && T < tmax) ? 1.f : 0.f;
            float spike_neg = (v < 0.f   && T > tmin) ? -1.f : 0.f;
            float spike     = spike_pos + spike_neg;

            float pf_mask = (t < max_pf_steps) ? 1.f : 0.f;
            v -= vthr * spike + pf_delta * pf_mask;

            T += spike;

            spike_seq_out[next] = spike;
            T_seq_out[next]     = T;
        }
        v_out[idx] = v;
    }
}

__global__ void forward_kernel_fp16(
    const __half* __restrict__ x_seq,
    const __half* __restrict__ v_th,
    const __half* __restrict__ T_max,
    const __half* __restrict__ T_min,
    const __half* __restrict__ prefire,
    __half* __restrict__ spike_seq_out,
    __half* __restrict__ v_out,
    __half* __restrict__ T_seq_out,
    __half* __restrict__ H_seq_out,
    int64_t time_steps, int64_t spatial)
{
    const int64_t stride = blockDim.x * gridDim.x;
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < spatial; idx += stride)
    {
        float vthr = __half2float(v_th[0]);
        float tmax = __half2float(T_max[0]);
        float tmin = __half2float(T_min[0]);
        float pf   = __half2float(prefire[0]);
        float v = (0.5f + pf) * vthr;
        float T = 0.0f;

        spike_seq_out[idx] = __float2half(0.0f);
        T_seq_out[idx]     = __float2half(0.0f);
        H_seq_out[idx]     = __float2half(v);

        const float pf_delta = (tmax > 0.f) ? (pf * vthr / tmax) : 0.f;
        const int   max_pf_steps = (int)fminf(fmaxf(tmax, 0.f), (float)time_steps);

        for (int64_t t = 0; t < time_steps; ++t) {
            const int64_t cur  = t * spatial + idx;
            const int64_t next = (t + 1) * spatial + idx;

            v += __half2float(x_seq[cur]);
            H_seq_out[next] = __float2half(v);

            float spike = 0.f;
            if (v >= vthr && T < tmax)      spike = 1.f;
            else if (v < 0.f && T > tmin)   spike = -1.f;

            float pf_mask = (t < max_pf_steps) ? 1.f : 0.f;
            v -= vthr * spike + pf_delta * pf_mask;

            T += spike;

            spike_seq_out[next] = __float2half(spike);
            T_seq_out[next]     = __float2half(T);
        }
        v_out[idx] = __float2half(v);
    }
}

__global__ void forward_kernel_fp64(
    const double* __restrict__ x_seq,
    const double* __restrict__ v_th,
    const double* __restrict__ T_max,
    const double* __restrict__ T_min,
    const double* __restrict__ prefire,
    double* __restrict__ spike_seq_out,
    double* __restrict__ v_out,
    double* __restrict__ T_seq_out,
    double* __restrict__ H_seq_out,
    int64_t time_steps, int64_t spatial)
{
    const int64_t stride = blockDim.x * gridDim.x;
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < spatial; idx += stride)
    {
        const double vthr = v_th[0];
        const double tmax = T_max[0];
        const double tmin = T_min[0];
        const double pf   = prefire[0];
        double v = (0.5 + pf) * vthr;
        double T = 0.0;

        spike_seq_out[idx] = 0.0;
        T_seq_out[idx]     = 0.0;
        H_seq_out[idx]     = v;

        const double pf_delta = (tmax > 0.0) ? (pf * vthr / tmax) : 0.0;
        const int64_t max_pf_steps = (int64_t)llrint(fmin(fmax(tmax, 0.0), (double)time_steps));

        for (int64_t t = 0; t < time_steps; ++t) {
            const int64_t cur  = t * spatial + idx;
            const int64_t next = (t + 1) * spatial + idx;

            v += x_seq[cur];
            H_seq_out[next] = v;

            double spike = 0.0;
            if (v >= vthr && T < tmax)      spike = 1.0;
            else if (v < 0.0 && T > tmin)   spike = -1.0;

            double pf_mask = (t < max_pf_steps) ? 1.0 : 0.0;
            v -= vthr * spike + pf_delta * pf_mask;

            T += spike;

            spike_seq_out[next] = spike;
            T_seq_out[next]     = T;
        }
        v_out[idx] = v;
    }
}

// ===================== Backward kernels =====================

__global__ void backward_kernel_fp32(
    const float* __restrict__ grad_spike_seq, // [T,   B*F]
    const float* __restrict__ grad_v,         // [B*F]
    const float* __restrict__ grad_T_seq,     // [T,   B*F]
    const float* __restrict__ spike_seq,      // [T+1, B*F]
    const float* __restrict__ T_seq,          // [T+1, B*F]
    const float* __restrict__ H_seq,          // [T+1, B*F]
    const float* __restrict__ v_th,
    const float* __restrict__ T_max,
    const float* __restrict__ T_min,
    float* __restrict__ grad_x_seq,           // [T,   B*F]
    int64_t time_steps, int64_t spatial)
{
    const float vthr = v_th[0];
    const float tmax = T_max[0];
    const float tmin = T_min[0];

    const int64_t stride = blockDim.x * gridDim.x;
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < spatial; idx += stride)
    {
        float gV = grad_v[idx];
        float gT = 0.0f;

        for (int64_t t = time_steps; t >= 1; --t) {
            const int64_t cur   = t * spatial + idx;       // H_t, T_t
            const int64_t prev  = (t - 1) * spatial + idx; // T_{t-1}
            const int64_t gout  = (t - 1) * spatial + idx; // grad_x[t-1]

            const float H_t   = H_seq[cur];
            const float T_t_1 = T_seq[prev];
            const float gY_t  = grad_spike_seq[gout];

            float dHdX = dtheta_gauss_f32(H_t - vthr, vthr, T_t_1, tmin, tmax) * theta_fp32(tmax - T_t_1)
                       + dtheta_gauss_f32(-H_t,        vthr, T_t_1, tmin, tmax) * theta_fp32(T_t_1 - tmin);

            float dYdT = -(thetaeq_fp32(H_t - vthr) * dtheta_gauss_f32(tmax - T_t_1, 1.0f, T_t_1, tmin, tmax)
                         + theta_fp32(-H_t)          * dtheta_gauss_f32(T_t_1 - tmin, 1.0f, T_t_1, tmin, tmax));

            float tmp  = gY_t - vthr * gV + gT;
            float gX   = tmp * dHdX + gV;
            gT         = tmp * dYdT + gT + grad_T_seq[prev];
            gV         = gX;

            grad_x_seq[gout] = gX;
        }
    }
}

__global__ void backward_kernel_fp16(
    const __half* __restrict__ grad_spike_seq,
    const __half* __restrict__ grad_v,
    const __half* __restrict__ grad_T_seq,
    const __half* __restrict__ spike_seq,
    const __half* __restrict__ T_seq,
    const __half* __restrict__ H_seq,
    const __half* __restrict__ v_th,
    const __half* __restrict__ T_max,
    const __half* __restrict__ T_min,
    __half* __restrict__ grad_x_seq,
    int64_t time_steps, int64_t spatial)
{
    const float vthr = __half2float(v_th[0]);
    const float tmax = __half2float(T_max[0]);
    const float tmin = __half2float(T_min[0]);

    const int64_t stride = blockDim.x * gridDim.x;
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < spatial; idx += stride)
    {
        float gV = __half2float(grad_v[idx]);
        float gT = 0.0f;

        for (int64_t t = time_steps; t >= 1; --t) {
            const int64_t cur   = t * spatial + idx;
            const int64_t prev  = (t - 1) * spatial + idx;
            const int64_t gout  = (t - 1) * spatial + idx;

            float H_t   = __half2float(H_seq[cur]);
            float T_t_1 = __half2float(T_seq[prev]);
            float gY_t  = __half2float(grad_spike_seq[gout]);
            float gT_t  = __half2float(grad_T_seq[prev]);

            float dHdX = dtheta_gauss_f32(H_t - vthr, vthr, T_t_1, tmin, tmax) * theta_fp32(tmax - T_t_1)
                       + dtheta_gauss_f32(-H_t,        vthr, T_t_1, tmin, tmax) * theta_fp32(T_t_1 - tmin);

            float dYdT = -(thetaeq_fp32(H_t - vthr) * dtheta_gauss_f32(tmax - T_t_1, 1.0f, T_t_1, tmin, tmax)
                         + theta_fp32(-H_t)          * dtheta_gauss_f32(T_t_1 - tmin, 1.0f, T_t_1, tmin, tmax));

            float tmp  = gY_t - vthr * gV + gT;
            float gX   = tmp * dHdX + gV;
            gT         = tmp * dYdT + gT + gT_t;
            gV         = gX;

            grad_x_seq[gout] = __float2half(gX);
        }
    }
}

__global__ void backward_kernel_fp64(
    const double* __restrict__ grad_spike_seq,
    const double* __restrict__ grad_v,
    const double* __restrict__ grad_T_seq,
    const double* __restrict__ spike_seq,
    const double* __restrict__ T_seq,
    const double* __restrict__ H_seq,
    const double* __restrict__ v_th,
    const double* __restrict__ T_max,
    const double* __restrict__ T_min,
    double* __restrict__ grad_x_seq,
    int64_t time_steps, int64_t spatial)
{
    const double vthr = v_th[0];
    const double tmax = T_max[0];
    const double tmin = T_min[0];

    const int64_t stride = blockDim.x * gridDim.x;
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < spatial; idx += stride)
    {
        double gV = grad_v[idx];
        double gT = 0.0;

        for (int64_t t = time_steps; t >= 1; --t) {
            const int64_t cur   = t * spatial + idx;
            const int64_t prev  = (t - 1) * spatial + idx;
            const int64_t gout  = (t - 1) * spatial + idx;

            const double H_t   = H_seq[cur];
            const double T_t_1 = T_seq[prev];
            const double gY_t  = grad_spike_seq[gout];

            double dHdX = dtheta_gauss_f64(H_t - vthr, vthr, T_t_1, tmin, tmax) * theta_fp64(tmax - T_t_1)
                        + dtheta_gauss_f64(-H_t,        vthr, T_t_1, tmin, tmax) * theta_fp64(T_t_1 - tmin);

            double dYdT = -(thetaeq_fp64(H_t - vthr) * dtheta_gauss_f64(tmax - T_t_1, 1.0, T_t_1, tmin, tmax)
                          + theta_fp64(-H_t)          * dtheta_gauss_f64(T_t_1 - tmin, 1.0, T_t_1, tmin, tmax));

            double tmp  = gY_t - vthr * gV + gT;
            double gX   = tmp * dHdX + gV;
            gT          = tmp * dYdT + gT + grad_T_seq[prev];
            gV          = gX;

            grad_x_seq[gout] = gX;
        }
    }
}

// ===================== Host wrappers =====================

static inline void check_inputs(const at::Tensor& x) {
    TORCH_CHECK(x.is_cuda(), "Tensor must be CUDA");
    TORCH_CHECK(x.is_contiguous(), "Tensor must be contiguous");
}

static inline std::tuple<int64_t,int64_t> get_time_and_spatial(const at::Tensor& x_seq) {
    const int64_t T = x_seq.size(0);
    // spatial = numel of one time-slice
    const int64_t spatial = x_seq[0].numel();
    return {T, spatial};
}

static inline std::vector<int64_t> make_out_sizes_Tp1(const at::Tensor& x_seq) {
    auto sizes = x_seq.sizes().vec();
    sizes[0] += 1; // T+1
    return sizes;
}

static inline std::vector<int64_t> make_out_sizes_T(const at::Tensor& x_seq) {
    auto sizes = x_seq.sizes().vec();
    sizes[0] = x_seq.size(0); // T
    return sizes;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
st_bif_forward_cuda(const at::Tensor& x_seq,
                    const at::Tensor& v_th,
                    const at::Tensor& T_max,
                    const at::Tensor& T_min,
                    const at::Tensor& prefire)
{
    check_inputs(x_seq);
    check_inputs(v_th); check_inputs(T_max); check_inputs(T_min); check_inputs(prefire);

    auto stream = at::cuda::getCurrentCUDAStream();
    const auto dtype = x_seq.scalar_type();
    const auto device = x_seq.device();

    auto [T, spatial] = get_time_and_spatial(x_seq);
    auto spike_all = at::empty(make_out_sizes_Tp1(x_seq), x_seq.options());
    auto T_all     = at::empty_like(spike_all);
    auto H_all     = at::empty_like(spike_all);
    auto v_out     = at::empty(x_seq[0].sizes(), x_seq.options());

    const int threads = 256;
    const int sm = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    const int blocks = std::min<int64_t>( (spatial + threads - 1) / threads, sm * 8 );

    if (dtype == at::kFloat) {
        forward_kernel_fp32<<<blocks, threads, 0, stream>>>(
            x_seq.data_ptr<float>(),
            v_th.data_ptr<float>(),
            T_max.data_ptr<float>(),
            T_min.data_ptr<float>(),
            prefire.data_ptr<float>(),
            spike_all.data_ptr<float>(),
            v_out.data_ptr<float>(),
            T_all.data_ptr<float>(),
            H_all.data_ptr<float>(),
            T, spatial
        );
    } else if (dtype == at::kHalf) {
        forward_kernel_fp16<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const __half*>(x_seq.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(v_th.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(T_max.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(T_min.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(prefire.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(spike_all.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(v_out.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(T_all.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(H_all.data_ptr<at::Half>()),
            T, spatial
        );
    } else if (dtype == at::kDouble) {
        forward_kernel_fp64<<<blocks, threads, 0, stream>>>(
            x_seq.data_ptr<double>(),
            v_th.data_ptr<double>(),
            T_max.data_ptr<double>(),
            T_min.data_ptr<double>(),
            prefire.data_ptr<double>(),
            spike_all.data_ptr<double>(),
            v_out.data_ptr<double>(),
            T_all.data_ptr<double>(),
            H_all.data_ptr<double>(),
            T, spatial
        );
    } else {
        TORCH_CHECK(false, "Unsupported dtype for forward");
    }

    CUDA_CHECK(cudaGetLastError());
    return {spike_all, v_out, T_all, H_all};
}

at::Tensor
st_bif_backward_cuda(const at::Tensor& grad_spike_seq,
                     const at::Tensor& grad_v,
                     const at::Tensor& grad_T_seq,
                     const at::Tensor& spike_all,
                     const at::Tensor& T_all,
                     const at::Tensor& H_all,
                     const at::Tensor& v_th,
                     const at::Tensor& T_max,
                     const at::Tensor& T_min)
{
    check_inputs(grad_spike_seq); check_inputs(grad_v); check_inputs(grad_T_seq);
    check_inputs(spike_all); check_inputs(T_all); check_inputs(H_all);
    check_inputs(v_th); check_inputs(T_max); check_inputs(T_min);

    auto stream = at::cuda::getCurrentCUDAStream();
    const auto dtype = grad_spike_seq.scalar_type();

    auto grad_x = at::empty_like(grad_spike_seq);

    auto [T, spatial] = get_time_and_spatial(spike_all) ; // spike_all: T+1
    const int64_t time_steps = T - 1;

    const int threads = 256;
    const int sm = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    const int blocks = std::min<int64_t>( (spatial + threads - 1) / threads, sm * 8 );

    if (dtype == at::kFloat) {
        backward_kernel_fp32<<<blocks, threads, 0, stream>>>(
            grad_spike_seq.data_ptr<float>(),
            grad_v.data_ptr<float>(),
            grad_T_seq.data_ptr<float>(),
            spike_all.data_ptr<float>(),
            T_all.data_ptr<float>(),
            H_all.data_ptr<float>(),
            v_th.data_ptr<float>(),
            T_max.data_ptr<float>(),
            T_min.data_ptr<float>(),
            grad_x.data_ptr<float>(),
            time_steps, spatial
        );
    } else if (dtype == at::kHalf) {
        backward_kernel_fp16<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const __half*>(grad_spike_seq.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(grad_v.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(grad_T_seq.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(spike_all.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(T_all.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(H_all.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(v_th.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(T_max.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(T_min.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(grad_x.data_ptr<at::Half>()),
            time_steps, spatial
        );
    } else if (dtype == at::kDouble) {
        backward_kernel_fp64<<<blocks, threads, 0, stream>>>(
            grad_spike_seq.data_ptr<double>(),
            grad_v.data_ptr<double>(),
            grad_T_seq.data_ptr<double>(),
            spike_all.data_ptr<double>(),
            T_all.data_ptr<double>(),
            H_all.data_ptr<double>(),
            v_th.data_ptr<double>(),
            T_max.data_ptr<double>(),
            T_min.data_ptr<double>(),
            grad_x.data_ptr<double>(),
            time_steps, spatial
        );
    } else {
        TORCH_CHECK(false, "Unsupported dtype for backward");
    }

    CUDA_CHECK(cudaGetLastError());
    return grad_x;
}

// ===================== Registration =====================

TORCH_LIBRARY(snn, m) {
    m.def("st_bif_forward(Tensor x_seq, Tensor v_th, Tensor T_max, Tensor T_min, Tensor prefire)"
         " -> (Tensor, Tensor, Tensor, Tensor)");
    m.def("st_bif_backward(Tensor grad_spike_seq, Tensor grad_v, Tensor grad_T_seq,"
         " Tensor spike_all, Tensor T_all, Tensor H_all, Tensor v_th, Tensor T_max, Tensor T_min)"
         " -> Tensor");
}

TORCH_LIBRARY_IMPL(snn, CUDA, m) {
    m.impl("st_bif_forward", st_bif_forward_cuda);
    m.impl("st_bif_backward", st_bif_backward_cuda);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
