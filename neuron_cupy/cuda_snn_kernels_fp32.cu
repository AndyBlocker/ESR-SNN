// cuda_snn_kernels.cu
extern "C" {

__device__ float theta(float x) {
    return x > 0.0f ? 1.0f : 0.0f;
}

__device__ float theta_eq(float x) {
    return x >= 0.0f ? 1.0f : 0.0f;
}

__device__ float theta_backward(float x) {
    float sigmoid = 1.0f / (1.0f + expf(-4.0f * x));
    return 4.0f * sigmoid * (1.0f - sigmoid);
}

__global__ void forward_kernel(
    const float* x_seq,
    const float* v_th,
    const float* T_max,
    const float* T_min,
    float* spike_seq_out,
    float* v_out,
    float* T_seq_out,
    float* H_seq_out,  // Added H_seq output
    int batch_size,
    int time_steps,
    int features
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * features) return;
    
    int b = idx / features;
    int f = idx % features;
    
    float v = 0.5f * v_th[0];
    float T = 0.0f;
    
    // Store initial states
    T_seq_out[idx] = T;
    spike_seq_out[idx] = 0.0f;
    H_seq_out[idx] = v;  // Store initial membrane potential
    
    for (int t = 0; t < time_steps; t++) {
        int current_idx = (t * batch_size * features) + idx;
        int next_idx = ((t + 1) * batch_size * features) + idx;
        
        // Update membrane potential
        v += x_seq[current_idx];
        H_seq_out[next_idx] = v;  // Store H_seq before spike update
        
        // Compute spike
        float spike = 0.0f;
        if (v >= v_th[0] && T < T_max[0]) {
            spike = 1.0f;
        } else if (v < 0.0f && T > T_min[0]) {
            spike = -1.0f;
        }
        
        // Update states
        v -= v_th[0] * spike;
        T += spike;
        
        // Store outputs
        spike_seq_out[next_idx] = spike;
        T_seq_out[next_idx] = T;
    }
    
    // Store final membrane potential
    v_out[idx] = v;
}

__global__ void backward_kernel(
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
    float grad_T = 0.0f;
    
    // Corrected loop bounds: from time_steps to 1
    for (int t = time_steps; t >= 1; t--) {
        int current_idx = (t * batch_size * features) + idx;
        int prev_idx = ((t-1) * batch_size * features) + idx;  // For accessing T_{t-1}
        
        float H_t = H_seq[current_idx];
        float T_t_1 = T_seq[prev_idx];  // Corrected indexing for T_{t-1}
        float grad_Y_t = grad_spike_seq[current_idx - batch_size * features];  // Adjust index for grad_spike_seq
        
        float grad_T_t_to_H_t = theta_backward(H_t - v_th[0]) * theta(T_max[0] - T_t_1) +
                               theta_backward(-H_t) * theta(T_t_1 - T_min[0]);
        
        float grad_Y_t_to_T_t_1 = -(theta_eq(H_t - v_th[0]) * theta_backward(T_max[0] - T_t_1) +
                                   theta(-H_t) * theta_backward(T_t_1 - T_min[0]));
        
        float grad_X = (grad_Y_t - v_th[0] * grad_V + grad_T) * grad_T_t_to_H_t + grad_V;
        grad_T = (grad_Y_t - v_th[0] * grad_V + grad_T) * grad_Y_t_to_T_t_1 + grad_T;
        grad_V = grad_X;
        
        grad_x_seq[current_idx - batch_size * features] = grad_X;  // Adjust index for output
    }
}

}