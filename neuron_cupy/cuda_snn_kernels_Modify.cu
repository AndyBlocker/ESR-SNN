#include <cuda_runtime.h>
#include <cuda_fp16.h>
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

__device__ float theta_backward_fp32(float x) {
    float sigmoid = float(1.0) / (float(1.0) + exp(float(-4.0) * x));
    return float(4.0) * sigmoid * (float(1.0) - sigmoid);
}

__device__ __half theta_backward_fp16(__half x) {
    __half sigmoid = __half(1.0) / __float2half(1.0f + (exp(__half(-4.0) * x)));
    return __half(4.0) * sigmoid * (__half(1.0) - sigmoid);
}

// FP32 kernels
__global__ void forward_kernel_fp32(
    const float* x_seq,
    const float* v_th,
    const float* T_max,
    const float* T_min,
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
        
    float v = 0.5f * v_th[0];
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
        
        v -= v_th[0] * spike;
        T += spike;
        
        spike_seq_out[next_idx] = spike;
        T_seq_out[next_idx] = T;
    }
    
    v_out[idx] = v;
}

// FP16 kernels
__global__ void forward_kernel_fp16(
    const __half* x_seq,
    const __half* v_th,
    const __half* T_max,
    const __half* T_min,
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
        
    __half v = __hmul(__float2half(0.5f), v_th[0]);
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
        
        v = __hsub(v, __hmul(v_th[0], spike));
        T = __hadd(T, spike);
        
        spike_seq_out[next_idx] = spike;
        T_seq_out[next_idx] = T;
    }
    
    v_out[idx] = v;
}

__global__ void backward_kernel_fp32(
    const float* grad_spike_seq,
    const float* grad_v,
    const float* grad_T_seq,
    const float* spike_seq,
    const float* T_seq,
    const float* H_seq,
    const float* v_th,
    const float* T_max,
    const float* T_min,
    float* grad_x_seq,
    int batch_size,
    int time_steps,
    int features
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * features) return;
    
    float grad_V = 0.0f;
    // float grad_T = 0.0f;
    
    // Corrected loop bounds: from time_steps to 1
    for (int t = time_steps; t >= 1; t--) {
        int current_idx = (t * batch_size * features) + idx;
        int prev_idx = ((t-1) * batch_size * features) + idx;  // For accessing T_{t-1}
        
        float H_t = H_seq[current_idx];
        float T_t_1 = T_seq[prev_idx];  // Corrected indexing for T_{t-1}
        float grad_Y_t = grad_spike_seq[current_idx - batch_size * features];  // Adjust index for grad_spike_seq
        
        float grad_Y_t_to_H_t = theta_backward_fp32(H_t - v_th[0]) * theta_fp32(T_max[0] - T_t_1) +
                               theta_backward_fp32(-H_t) * theta_fp32(T_t_1 - T_min[0]);
        
        // float grad_Y_t_to_T_t_1 = -(theta_eq_fp32(H_t - v_th[0]) * theta_backward_fp32(T_max[0] - T_t_1) +
        //                            theta_fp32(-H_t) * theta_backward_fp32(T_t_1 - T_min[0]));
        float grad_X = 0.0f;
        if(t == time_steps)
            grad_X = grad_Y_t*grad_Y_t_to_H_t + (grad_v[idx] + grad_V) * (1 - v_th[0]*grad_Y_t_to_H_t);
        else
            grad_X = grad_Y_t*grad_Y_t_to_H_t + grad_V * (1 - v_th[0]*grad_Y_t_to_H_t);
        // float grad_X = (grad_Y_t - v_th[0] * grad_V + grad_T) * grad_T_t_to_H_t + grad_V;
        // grad_T = (grad_Y_t - v_th[0] * grad_V + grad_T) * grad_Y_t_to_T_t_1 + grad_T;
        grad_V = grad_X;
        
        grad_x_seq[current_idx - batch_size * features] = grad_X;  // Adjust index for output
    }
}


__global__ void backward_kernel_fp16(
    const __half* grad_spike_seq,
    const __half* grad_v,
    const __half* grad_T_seq,
    const __half* spike_seq,
    const __half* T_seq,
    const __half* H_seq,
    const __half* v_th,
    const __half* T_max,
    const __half* T_min,
    __half* grad_x_seq,
    int batch_size,
    int time_steps,
    int features
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * features) return;
    
    __half grad_V = __float2half(0.0f);
    // __half grad_T = __float2half(0.0f);
    
    for (int t = time_steps; t >= 1; t--) {
        int current_idx = (t * batch_size * features) + idx;
        int prev_idx = ((t-1) * batch_size * features) + idx;
        
        __half H_t = H_seq[current_idx];
        __half T_t_1 = T_seq[prev_idx];
        __half grad_Y_t = grad_spike_seq[current_idx - batch_size * features];
        
        __half grad_Y_t_to_H_t = __hadd(__hmul(theta_backward_fp16(__hsub(H_t, v_th[0])), theta_fp16(__hsub(T_max[0], T_t_1))), \
                                        __hmul(theta_backward_fp16(__hneg(H_t)), theta_fp16(__hsub(T_t_1, T_min[0]))));

        // __half grad_Y_t_to_T_t_1 = __hneg(__hadd(__hmul(theta_eq_fp16(__hsub(H_t, v_th[0])), theta_backward_fp16(__hsub(T_max[0], T_t_1))) ,\
        //                                          __hmul(theta_fp16(__hneg(H_t)), theta_backward_fp16(__hsub(T_t_1, T_min[0])))));

        // __half tmp = __hadd(__hsub(grad_Y_t,__hmul(v_th[0],grad_V)), grad_T);
        // __half grad_X = __hadd(__hmul(tmp,grad_T_t_to_H_t), grad_V);
        // grad_T = __hadd(__hmul(tmp,grad_Y_t_to_T_t_1), grad_T);
        __half grad_X = __float2half(0.0f);
        if(t == time_steps)
            grad_X = __hadd(__hmul(grad_Y_t, grad_Y_t_to_H_t), __hmul(__hadd(grad_V,grad_v[idx]),__hsub(__float2half(1.0f), __hmul(v_th[0],grad_Y_t_to_H_t))));
        else
            grad_X = __hadd(__hmul(grad_Y_t, grad_Y_t_to_H_t), __hmul(grad_V,__hsub(__float2half(1.0f), __hmul(v_th[0],grad_Y_t_to_H_t))));
        grad_V = grad_X;

        // Store result
        grad_x_seq[current_idx - batch_size * features] = grad_X;
    }
}

}