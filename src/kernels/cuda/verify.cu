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
    extern __shared__ float shared_data[];
    float* shared_max = shared_data;
    int32_t* shared_argmax = (int32_t*)(shared_data + K * 32);
    
    // Process each K position
    for (int k = 0; k < K; k++) {
        if (tid < 32) {  // One warp per K
            float max_val = -INFINITY;
            int32_t argmax_idx = 0;
            
            // Find argmax in logits[batch_idx][k][:]
            for (int v = lane_id; v < V; v += 32) {
                float val = logits[batch_idx * K * V + k * V + v];
                if (val > max_val) {
                    max_val = val;
                    argmax_idx = v;
                }
            }
            
            // Warp-level reduction to find global argmax
            for (int offset = 16; offset > 0; offset /= 2) {
                float other_val = __shfl_down_sync(0xFFFFFFFF, max_val, offset);
                int32_t other_idx = __shfl_down_sync(0xFFFFFFFF, argmax_idx, offset);
                if (other_val > max_val || (other_val == max_val && other_idx < argmax_idx)) {
                    max_val = other_val;
                    argmax_idx = other_idx;
                }
            }
            
            // Store results in shared memory
            if (lane_id == 0) {
                shared_max[k] = max_val;
                shared_argmax[k] = argmax_idx;
            }
        }
        __syncthreads();
    }
    
    // Check matches and compute prefix length
    if (tid == 0) {
        int32_t prefix_len = 0;
        int32_t target_id = draft_ids[batch_idx * K];
        
        // Check if first position matches
        if (shared_argmax[0] == target_id) {
            prefix_len = 1;
            accepted_mask[batch_idx * K] = 1;
            
            // Check subsequent positions
            for (int k = 1; k < K; k++) {
                target_id = draft_ids[batch_idx * K + k];
                if (shared_argmax[k] == target_id) {
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
    dim3 block(32);  // One warp per block
    dim3 grid(B);    // One block per batch item
    size_t shared_mem_size = K * 32 * sizeof(float) + K * 32 * sizeof(int32_t);
    
    verify_prefix_kernel<<<grid, block, shared_mem_size>>>(
        logits.data_ptr<float>(),
        draft_ids.data_ptr<int32_t>(),
        accept_len.data_ptr<int32_t>(),
        accepted_mask.data_ptr<uint8_t>(),
        B, K, V
    );
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(err)));
    }
    
    return {accept_len, accepted_mask};
}

// Python binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("verify_prefix", &verify_prefix_cuda, "Verify prefix matches (CUDA)");
}
