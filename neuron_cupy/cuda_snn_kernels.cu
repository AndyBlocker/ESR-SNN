#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
// cuda_snn_kernels.cu
extern "C" {

__device__ float theta_fp32(float x) {
return x > float(0.0) ? float(1.0) : float(0.0);
}

__device__ __half theta_fp16(__half x) {
return x > __half(0.0) ? __half(1.0) : __half(0.0);
}

__device__ float theta_eq_fp32(float x) {
return x >= float(0.0) ? float(1.0) : float(0.0);
}

__device__ __half theta_eq_fp16(__half x) {
return x >= __half(0.0) ? __half(1.0) : __half(0.0);
}

// softsgn
// __device__ float theta_backward_fp32(float x, float V_thr, float S, float S_min, float S_max) {
//     float scale = 2.0f/V_thr;
//     float sigmoid = float(1.0) + fabsf(scale * x);
//     return scale / (2 * sigmoid * sigmoid);
// }

// __device__ __half theta_backward_fp16(__half x, __half V_thr, __half S, __half S_min, __half S_max) {
//     __half scale = __hdiv(__float2half(2.0f), V_thr);
//     __half sigmoid = __hadd(__float2half(1.0f), __habs(__hmul(scale, x)));
//     return __hdiv(scale, __hmul(__float2half(2.0f),__hmul(sigmoid, sigmoid)));
// }

// // Atan
// __device__ float theta_backward_fp32(float x, float V_thr, float S, float S_min, float S_max) {
//     float scale = 2.0f/V_thr;
//     float half_pi = 1.57079632679489661923f;
//     float sigmoid = (half_pi * scale * x);
//     return scale / (2 * (1 + sigmoid * sigmoid));
// }

// __device__ __half theta_backward_fp16(__half x, __half V_thr, __half S, __half S_min, __half S_max) {
//     __half scale = __hdiv(__float2half(2.0f), V_thr);
//     __half half_pi = 1.57079632679489661923f;
//     __half sigmoid = __hmul(half_pi, (__hmul(scale, x)));
//     return __hdiv(scale, __hmul(__float2half(2.0f),__hadd(__float2half(1.0f),__hmul(sigmoid, sigmoid))));
// }


// sigmoid
__device__ float theta_backward_fp32(float x, float V_thr, float S, float S_min, float S_max) {
float sigmoid = float(1.0) / (float(1.0) + exp(float(-4.0) * x / V_thr));
return float(4.0)/ V_thr * sigmoid * (float(1.0) - sigmoid);
}

__device__ __half theta_backward_fp16(__half x, __half V_thr, __half S, __half S_min, __half S_max) {
__half sigmoid = __hdiv(__float2half(1.0f),  __float2half(__hadd(__float2half(1.0f), __float2half(hexp(__hmul(__float2half(-4.0f), __hdiv(x,V_thr)))))));
return __hmul(__hdiv(__float2half(4.0f), V_thr), __hmul(sigmoid,(__hsub(__float2half(1.0f), sigmoid)))) ;
}

// normalized Gaussian
// __device__ float theta_backward_fp32(float x, float V_thr, float S, float S_min, float S_max) {
//     float mu = 0.0f;
//     float sqrt_2pi = 2.506628253;
//     float sigma = 1/sqrt_2pi*V_thr;
//     sigma = fmaxf(sigma, 1e-3f);
//     float a = 1.0f/(sigma*sqrt_2pi);
//     float upper_x = -(x - mu) * (x - mu) / (2.0f * sigma * sigma);
//     return a * exp(upper_x);
// }

// __device__ __half theta_backward_fp16(__half x, __half V_thr, __half S, __half S_min, __half S_max) {
//     float mu = 0.0f;
//     float sqrt_2pi = 2.506628253f;
//     float sigma = 1/sqrt_2pi*float(V_thr);
//     sigma = fmaxf(sigma, 1e-3f);
//     float a = 1.0f/(sigma*sqrt_2pi);
//     float upper_x = -(float(x) - mu) * (float(x) - mu) / (2.0f * sigma * sigma);
//     return __float2half(a * exp(upper_x));
// }


// __device__ __half theta_backward_fp16(__half x, __half V_thr, __half S, __half S_min, __half S_max) {
//     __half mu = __float2half(0.0f);
//     __half sigma = __hmul(__float2half(0.4f), V_thr);
//     __half sqrt_2pi = __float2half(2.506628253f);
//     __half a = __hdiv(__float2half(1.0f), __hmul(sigma, sqrt_2pi));
//     __half upper_x = __hdiv(__hmul(__hneg(__hsub(x, mu)), __hsub(x, mu)), __hmul(__float2half(2.0f), __hmul(sigma, sigma)));
//     return __hmul(a, hexp(upper_x));
// }

// __device__ float theta_backward_fp32(float x, float V_thr, float S, float S_min, float S_max) {
//     float mu = 0.0f;
//     float sigma = 0.4f * V_thr;
//     float a = 1.0f/sigma;
//     float upper_x = -(x - mu) * (x - mu) / (2.0f * sigma * sigma);
//     return a * exp(upper_x);
// }

// __device__ __half theta_backward_fp16(__half x, __half V_thr, __half S, __half S_min, __half S_max) {
//     __half mu = __float2half(0.0f);
//     __half sigma = __hmul(__float2half(0.4f), V_thr);
//     __half a = __hdiv(__float2half(1.0f), sigma);
//     __half upper_x = __hdiv(__hmul(__hneg(__hsub(x, mu)), __hsub(x, mu)), __hmul(__float2half(2.0f), __hmul(sigma, sigma)));
//     return __hmul(a, hexp(upper_x));
// }

// FP32 kernels
__global__ void forward_kernel_fp32(
const float* x_seq,
const float* v_th,
const float* T_max,
const float* T_min,
const float* prefire,
float* spike_seq_out,
float* v_out,
float* T_seq_out,
float* H_seq_out,
int batch_size,
int time_steps,
int features
) {
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx >= batch_size * features) return;
    
float v = (0.5f + prefire[0]) * v_th[0];
float T = 0.0f;

T_seq_out[idx] = T;
spike_seq_out[idx] = 0.0f;
H_seq_out[idx] = v;

for (int t = 0; t < time_steps; t++) {
    int current_idx = (t * batch_size * features) + idx;
    int next_idx = ((t + 1) * batch_size * features) + idx;
    
    v += x_seq[current_idx];
    H_seq_out[next_idx] = v;
    
    float spike = 0.0f;
    if (v >= v_th[0] && T < T_max[0]) {
        spike = 1.0f;
    } else if (v < 0.0f && T > T_min[0]) {
        spike = -1.0f;
    }
    
    if (t < T_max[0]){
        v -= (v_th[0] * spike + prefire[0]*v_th[0]/T_max[0]);
    }
    else{
        v -= (v_th[0] * spike);
    }

    T += spike;
    
    spike_seq_out[next_idx] = spike;
    T_seq_out[next_idx] = T;
    v_out[current_idx] = v;
}

}

// FP16 kernels
__global__ void forward_kernel_fp16(
const __half* x_seq,
const __half* v_th,
const __half* T_max,
const __half* T_min,
const __half* prefire,
__half* spike_seq_out,
__half* v_out,
__half* T_seq_out,
__half* H_seq_out,
int batch_size,
int time_steps,
int features
) {
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx >= batch_size * features) return;
    
__half v = __hmul(__hadd(__float2half(0.5f), prefire[0]), v_th[0]);
__half T = __float2half(0.0f);

T_seq_out[idx] = T;
spike_seq_out[idx] = __float2half(0.0f);
H_seq_out[idx] = v;

for (int t = 0; t < time_steps; t++) {
    int current_idx = (t * batch_size * features) + idx;
    int next_idx = ((t + 1) * batch_size * features) + idx;
    
    v = __hadd(v, x_seq[current_idx]);
    H_seq_out[next_idx] = v;
    
    __half spike = __float2half(0.0f);
    if (__hge(v, v_th[0]) && __hlt(T, T_max[0])) {
        spike = __float2half(1.0f);
    } else if (__hlt(v, __float2half(0.0f)) && __hgt(T, T_min[0])) {
        spike = __float2half(-1.0f);
    }
    
    if (__hlt(t, T_max[0])){
        v = __hsub(__hsub(v, __hmul(v_th[0], spike)), __hdiv(__hmul(prefire[0], v_th[0]), T_max[0]));
    }
    else{
        v = __hsub(v, __hmul(v_th[0], spike));
    }
    T = __hadd(T, spike);
    
    spike_seq_out[next_idx] = spike;
    T_seq_out[next_idx] = T;
    v_out[current_idx] = v;
}

}

__global__ void backward_kernel_fp32(
const float* grad_spike_seq,
const float* grad_v_seq,
const float* grad_T_seq,
const float* spike_seq,
const float* T_seq,
const float* H_seq,
const float* v_th,
const float* T_max,
const float* T_min,
float* grad_x_seq,
float* grad_V_th_cp,
int batch_size,
int time_steps,
int features
) {
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx >= batch_size * features) return;

float grad_V = 0.0f;
float grad_T = 0.0f;

// Corrected loop bounds: from time_steps to 1
for (int t = time_steps; t >= 1; t--) {
    int current_idx = (t * batch_size * features) + idx;
    int prev_idx = ((t-1) * batch_size * features) + idx;  // For accessing T_{t-1}
    
    float H_t = H_seq[current_idx];
    float T_t_1 = T_seq[prev_idx];  // Corrected indexing for T_{t-1}
    float grad_Y_t = grad_spike_seq[current_idx - batch_size * features];
    
    float grad_T_t_to_H_t = theta_backward_fp32(H_t - v_th[0], v_th[0], T_t_1, T_min[0], T_max[0]) * theta_fp32(T_max[0] - T_t_1) +
                            theta_backward_fp32(-H_t,v_th[0], T_t_1, T_min[0], T_max[0]) * theta_fp32(T_t_1 - T_min[0]);

    // float grad_T_t_to_H_t_max1 = theta_backward_fp32(-v_th[0])* theta_fp32(T_max[0] - T_t_1) + theta_backward_fp32(0.0f)* theta_fp32(T_t_1 - T_min[0]);
    
    // float grad_T_t_to_H_t_max2 = theta_backward_fp32(-v_th[0]/2)* theta_fp32(T_max[0] - T_t_1) + theta_backward_fp32(v_th[0]/2)* theta_fp32(T_t_1 - T_min[0]);

    // grad_T_t_to_H_t = grad_T_t_to_H_t/ fmaxf(grad_T_t_to_H_t_max1, grad_T_t_to_H_t_max2);

    float grad_Y_t_to_T_t_1 = -(theta_eq_fp32(H_t - v_th[0]) * theta_backward_fp32(T_max[0] - T_t_1,1.0f, T_t_1, T_min[0], T_max[0]) +
                                theta_fp32(-H_t) * theta_backward_fp32(T_t_1 - T_min[0],1.0f, T_t_1, T_min[0], T_max[0]));
    
    float grad_X = (grad_Y_t - v_th[0] * grad_V + grad_T) * grad_T_t_to_H_t + grad_V;
    grad_T = (grad_Y_t - v_th[0] * grad_V + grad_T) * grad_Y_t_to_T_t_1 + grad_T + grad_T_seq[prev_idx];
    grad_V = grad_X + grad_v_seq[prev_idx];
    
    grad_x_seq[current_idx - batch_size * features] = grad_X;  // Adjust index for output
    grad_V_th_cp[current_idx - batch_size * features] = (grad_T_seq[prev_idx] - v_th[0] * grad_V + grad_Y_t) * (-theta_backward_fp32(H_t - v_th[0], v_th[0], T_t_1, T_min[0], T_max[0]) * theta_fp32(T_max[0] - T_t_1));
}
}


__global__ void backward_kernel_fp16(
const __half* grad_spike_seq,
const __half* grad_v_seq,
const __half* grad_T_seq,
const __half* spike_seq,
const __half* T_seq,
const __half* H_seq,
const __half* v_th,
const __half* T_max,
const __half* T_min,
__half* grad_x_seq,
__half* grad_V_th_cp,
int batch_size,
int time_steps,
int features
) {
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx >= batch_size * features) return;

__half grad_V = __float2half(0.0f);
__half grad_T = __float2half(0.0f);

for (int t = time_steps; t >= 1; t--) {
    int current_idx = (t * batch_size * features) + idx;
    int prev_idx = ((t-1) * batch_size * features) + idx;
    
    __half H_t = H_seq[current_idx];
    __half T_t_1 = T_seq[prev_idx];
    __half grad_Y_t = grad_spike_seq[current_idx - batch_size * features];
    
    __half grad_T_t_to_H_t = __hadd(__hmul(theta_backward_fp16(__hsub(H_t, v_th[0]),v_th[0], T_t_1, T_min[0], T_max[0]), theta_fp16(__hsub(T_max[0], T_t_1))), \
                                    __hmul(theta_backward_fp16(__hneg(H_t),v_th[0], T_t_1, T_min[0], T_max[0]), theta_fp16(__hsub(T_t_1, T_min[0]))));

    // __half grad_T_t_to_H_t_max1 = __hadd(__hmul(theta_backward_fp16(__hneg(v_th[0])), theta_fp16(__hsub(T_max[0], T_t_1))), \
    //                                 __hmul(theta_backward_fp16(__float2half(0.0f)), theta_fp16(__hsub(T_t_1, T_min[0]))));

    // __half grad_T_t_to_H_t_max2 = __hadd(__hmul(theta_backward_fp16(__hmul(__float2half(0.5f), v_th[0])), theta_fp16(__hsub(T_max[0], T_t_1))), \
    //                                 __hmul(theta_backward_fp16(__hneg(__hmul(__float2half(0.5f), v_th[0]))), theta_fp16(__hsub(T_t_1, T_min[0]))));
    
    // grad_T_t_to_H_t = __hdiv(grad_T_t_to_H_t, __hmax(grad_T_t_to_H_t_max1, grad_T_t_to_H_t_max2));

    __half grad_Y_t_to_T_t_1 = __hneg(__hadd(__hmul(theta_eq_fp16(__hsub(H_t, v_th[0])), theta_backward_fp16(__hsub(T_max[0], T_t_1),__float2half(1.0f), T_t_1, T_min[0], T_max[0])) ,\
                                                __hmul(theta_fp16(__hneg(H_t)), theta_backward_fp16(__hsub(T_t_1, T_min[0]),__float2half(1.0f), T_t_1, T_min[0], T_max[0]))));


    __half grad_X = __hadd(__hmul(__hadd(__hsub(grad_Y_t,__hmul(v_th[0],grad_V)), grad_T),grad_T_t_to_H_t), grad_V);
    grad_T = __hadd(__hadd(__hmul(__hadd(__hsub(grad_Y_t,__hmul(v_th[0],grad_V)), grad_T),grad_Y_t_to_T_t_1), grad_T), grad_T_seq[prev_idx]);
    grad_V = __hadd(grad_X, grad_v_seq[prev_idx]);

    // Store result
    grad_x_seq[current_idx - batch_size * features] = grad_X;
    // grad_V_th_cp[current_idx - batch_size * features] = (grad_T_seq[prev_idx] - v_th[0] * grad_V + grad_Y_t) * (-theta_backward_fp32(H_t - v_th[0], v_th[0], T_t_1, T_min[0], T_max[0]) * theta_fp32(T_max[0] - T_t_1));
    grad_V_th_cp[current_idx - batch_size * features] = __hmul(__hadd(__hsub(grad_T_seq[prev_idx], __hmul(v_th[0], grad_V)), grad_Y_t), __hneg(__hmul(theta_backward_fp16(__hsub(H_t, v_th[0]), v_th[0], T_t_1, T_min[0], T_max[0]), theta_fp16(__hsub(T_max[0], T_t_1)))));
}
}


// BF16 kernels (compute in FP32 for correctness, store as BF16)
__global__ void forward_kernel_bf16(
const __nv_bfloat16* x_seq,
const __nv_bfloat16* v_th,
const __nv_bfloat16* T_max,
const __nv_bfloat16* T_min,
const __nv_bfloat16* prefire,
__nv_bfloat16* spike_seq_out,
__nv_bfloat16* v_out,
__nv_bfloat16* T_seq_out,
__nv_bfloat16* H_seq_out,
int batch_size,
int time_steps,
int features
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * features) return;

    float v_th0 = __bfloat162float(v_th[0]);
    float T_max0 = __bfloat162float(T_max[0]);
    float T_min0 = __bfloat162float(T_min[0]);
    float prefire0 = __bfloat162float(prefire[0]);

    float v = (0.5f + prefire0) * v_th0;
    float T = 0.0f;

    T_seq_out[idx] = __float2bfloat16(T);
    spike_seq_out[idx] = __float2bfloat16(0.0f);
    H_seq_out[idx] = __float2bfloat16(v);

    for (int t = 0; t < time_steps; t++) {
        int current_idx = (t * batch_size * features) + idx;
        int next_idx = ((t + 1) * batch_size * features) + idx;

        v += __bfloat162float(x_seq[current_idx]);
        H_seq_out[next_idx] = __float2bfloat16(v);

        float spike = 0.0f;
        if (v >= v_th0 && T < T_max0) {
            spike = 1.0f;
        } else if (v < 0.0f && T > T_min0) {
            spike = -1.0f;
        }

        if ((float)t < T_max0) {
            v -= (v_th0 * spike + prefire0 * v_th0 / T_max0);
        } else {
            v -= (v_th0 * spike);
        }

        T += spike;

        spike_seq_out[next_idx] = __float2bfloat16(spike);
        T_seq_out[next_idx] = __float2bfloat16(T);
        v_out[current_idx] = __float2bfloat16(v);
    }
}


__global__ void backward_kernel_bf16(
const __nv_bfloat16* grad_spike_seq,
const __nv_bfloat16* grad_v_seq,
const __nv_bfloat16* grad_T_seq,
const __nv_bfloat16* spike_seq,
const __nv_bfloat16* T_seq,
const __nv_bfloat16* H_seq,
const __nv_bfloat16* v_th,
const __nv_bfloat16* T_max,
const __nv_bfloat16* T_min,
__nv_bfloat16* grad_x_seq,
__nv_bfloat16* grad_V_th_cp,
int batch_size,
int time_steps,
int features
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * features) return;

    float v_th0 = __bfloat162float(v_th[0]);
    float T_max0 = __bfloat162float(T_max[0]);
    float T_min0 = __bfloat162float(T_min[0]);

    float grad_V = 0.0f;
    float grad_T = 0.0f;

    for (int t = time_steps; t >= 1; t--) {
        int current_idx = (t * batch_size * features) + idx;
        int prev_idx = ((t - 1) * batch_size * features) + idx;

        float H_t   = __bfloat162float(H_seq[current_idx]);
        float T_t_1 = __bfloat162float(T_seq[prev_idx]);
        float grad_Y_t = __bfloat162float(grad_spike_seq[current_idx - batch_size * features]);

        float grad_T_t_to_H_t =
            theta_backward_fp32(H_t - v_th0, v_th0, T_t_1, T_min0, T_max0) * theta_fp32(T_max0 - T_t_1) +
            theta_backward_fp32(-H_t,       v_th0, T_t_1, T_min0, T_max0) * theta_fp32(T_t_1 - T_min0);

        float grad_Y_t_to_T_t_1 =
            -(theta_eq_fp32(H_t - v_th0) * theta_backward_fp32(T_max0 - T_t_1, 1.0f, T_t_1, T_min0, T_max0) +
              theta_fp32(-H_t)          * theta_backward_fp32(T_t_1 - T_min0, 1.0f, T_t_1, T_min0, T_max0));

        float grad_X = (grad_Y_t - v_th0 * grad_V + grad_T) * grad_T_t_to_H_t + grad_V;

        grad_T = (grad_Y_t - v_th0 * grad_V + grad_T) * grad_Y_t_to_T_t_1 + grad_T +
                 __bfloat162float(grad_T_seq[prev_idx]);

        grad_V = grad_X + __bfloat162float(grad_v_seq[prev_idx]);

        grad_x_seq[current_idx - batch_size * features] = __float2bfloat16(grad_X);

        float grad_vth =
            (grad_Y_t * theta_eq_fp32(H_t - v_th0) *
             theta_backward_fp32(H_t - v_th0, v_th0, T_t_1, T_min0, T_max0) *
             theta_fp32(T_max0 - T_t_1));

        grad_V_th_cp[current_idx - batch_size * features] = __float2bfloat16(grad_vth);
    }
}

}
