#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

// CUDA kernel for verify_prefix
// Inputs: logits [B][K][V], draft_ids [B][K]
// Outputs: accept_len [B], accepted_mask [B][K]
// Optimized for small K (<=8) with one block per batch item

__global__ void verify_prefix_kernel(
    const float* logits,           // [B][K][V]
    const int32_t* draft_ids,      // [B][K]
    int32_t* accept_len,           // [B]
    uint8_t* accepted_mask,        // [B][K]
    int B, int K, int V
) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= B) return;
    
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    
    // Shared memory for reduction
    // OPTIMIZATION: Allocate shared memory dynamically based on block size
    // Layout: [reduction buffer: blockDim.x floats + blockDim.x int32s] [final results: K floats + K int32s]
    extern __shared__ float shared_data[];
    float* reduction_max = shared_data;
    int32_t* reduction_argmax = (int32_t*)(shared_data + blockDim.x);
    float* final_max = (float*)(reduction_argmax + blockDim.x);
    int32_t* final_argmax = (int32_t*)(final_max + K);
    
    // Process each K position
    // OPTIMIZATION: Use multiple warps for better parallelism on large vocabularies
    for (int k = 0; k < K; k++) {
        float max_val = -INFINITY;
        int32_t argmax_idx = 0;
        
        // Find argmax in logits[batch_idx][k][:] using all threads
        // Each thread processes V/blockDim.x elements
        for (int v = tid; v < V; v += blockDim.x) {
            float val = logits[batch_idx * K * V + k * V + v];
            if (val > max_val) {
                max_val = val;
                argmax_idx = v;
            }
        }
        
        // Store thread-local max in shared memory reduction buffer
        reduction_max[tid] = max_val;
        reduction_argmax[tid] = argmax_idx;
        __syncthreads();
        
        // Reduction across threads in block to find global argmax
        // Use binary reduction tree for efficiency
        for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
            if (tid < stride) {
                int other_idx = tid + stride;
                if (other_idx < blockDim.x) {
                    float other_val = reduction_max[other_idx];
                    int32_t other_argmax = reduction_argmax[other_idx];
                    if (other_val > reduction_max[tid] || 
                        (other_val == reduction_max[tid] && other_argmax < reduction_argmax[tid])) {
                        reduction_max[tid] = other_val;
                        reduction_argmax[tid] = other_argmax;
                    }
                }
            }
            __syncthreads();
        }
        
        // Store final result for this K position
        if (tid == 0) {
            final_max[k] = reduction_max[0];
            final_argmax[k] = reduction_argmax[0];
        }
        __syncthreads();
    }
    
    // Check matches and compute prefix length
    // Use final results stored in shared memory
    if (tid == 0) {
        int32_t prefix_len = 0;
        int32_t target_id = draft_ids[batch_idx * K];
        
        // Check if first position matches
        if (final_argmax[0] == target_id) {
            prefix_len = 1;
            accepted_mask[batch_idx * K] = 1;
            
            // Check subsequent positions
            for (int k = 1; k < K; k++) {
                target_id = draft_ids[batch_idx * K + k];
                if (final_argmax[k] == target_id) {
                    prefix_len++;
                    accepted_mask[batch_idx * K + k] = 1;
                } else {
                    // Fill remaining positions with 0
                    for (int remaining = k; remaining < K; remaining++) {
                        accepted_mask[batch_idx * K + remaining] = 0;
                    }
                    break;
                }
            }
        } else {
            // No matches, set all to 0
            for (int k = 0; k < K; k++) {
                accepted_mask[batch_idx * K + k] = 0;
            }
        }
        
        accept_len[batch_idx] = prefix_len;
    }
}

// Wrapper function
std::vector<torch::Tensor> verify_prefix_cuda(
    torch::Tensor logits,
    torch::Tensor draft_ids
) {
    // Input validation
    TORCH_CHECK(logits.dim() == 3, "logits must be 3D tensor [B][K][V]");
    TORCH_CHECK(draft_ids.dim() == 2, "draft_ids must be 2D tensor [B][K]");
    TORCH_CHECK(logits.size(0) == draft_ids.size(0), "Batch size mismatch");
    TORCH_CHECK(logits.size(1) == draft_ids.size(1), "K dimension mismatch");
    TORCH_CHECK(logits.device().is_cuda(), "logits must be on CUDA");
    TORCH_CHECK(draft_ids.device().is_cuda(), "draft_ids must be on CUDA");
    
    int B = logits.size(0);
    int K = logits.size(1);
    int V = logits.size(2);
    
    // Create output tensors
    auto accept_len = torch::zeros({B}, torch::TensorOptions().dtype(torch::kInt32).device(logits.device()));
    auto accepted_mask = torch::zeros({B, K}, torch::TensorOptions().dtype(torch::kUInt8).device(logits.device()));
    
    // Launch kernel
    // OPTIMIZATION: Use 256 threads (8 warps) per block for better GPU utilization
    // This allows better parallelism for large vocabularies (V > 10K)
    // For small vocabularies, the overhead is minimal, but for large ones (50K+), this is significant
    int threads_per_block = 256;  // 8 warps for better parallelism
    dim3 block(threads_per_block);
    dim3 grid(B);    // One block per batch item
    // Shared memory: threads_per_block floats + threads_per_block int32s for reduction, plus K floats + K int32s for final results
    // CRITICAL FIX: Use threads_per_block (host variable) instead of blockDim.x (device variable)
    size_t shared_mem_size = threads_per_block * sizeof(float) + threads_per_block * sizeof(int32_t) + 
                             K * sizeof(float) + K * sizeof(int32_t);
    
    verify_prefix_kernel<<<grid, block, shared_mem_size>>>(
        logits.data_ptr<float>(),
        draft_ids.data_ptr<int32_t>(),
        accept_len.data_ptr<int32_t>(),
        accepted_mask.data_ptr<uint8_t>(),
        B, K, V
    );
    
    // Check for launch errors immediately
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(err)));
    }
    
    // OPTIMIZATION: For production code, we could add cudaDeviceSynchronize() here to catch execution errors
    // However, for async execution with streams, we skip this to allow overlap
    // The caller should synchronize when needed
    
    return {accept_len, accepted_mask};
}

// Python binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("verify_prefix", &verify_prefix_cuda, "Verify prefix matches (CUDA)");
}
