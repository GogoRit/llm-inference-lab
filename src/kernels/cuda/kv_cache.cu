#include <torch/extension.h>
#include <cuda_runtime.h>

// CUDA kernel for KV cache append
// Appends K accepted key/value blocks from draft KV to base KV at target offset
// Optimized for coalesced loads/stores

template<typename T>
__global__ void kv_append_kernel(
    const T* base_k,           // [B][H][L][D] - base key cache
    const T* base_v,           // [B][H][L][D] - base value cache
    const T* draft_k,          // [B][H][K][D] - draft key cache
    const T* draft_v,          // [B][H][K][D] - draft value cache
    T* output_k,               // [B][H][L+K][D] - output key cache
    T* output_v,               // [B][H][L+K][D] - output value cache
    const uint8_t* accepted_mask, // [B][K] - which draft positions to append
    const int32_t* accept_len,    // [B] - how many to append per batch
    int B, int H, int L, int K, int D,
    int offset                 // where to start appending in base cache
) {
    int batch_idx = blockIdx.y;
    int head_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    if (batch_idx >= B || head_idx >= H) return;
    
    int num_accepted = accept_len[batch_idx];
    if (num_accepted == 0) return;
    
    // Copy base cache first
    int base_size = L * D;
    int base_offset = batch_idx * H * L * D + head_idx * L * D;
    int output_base_offset = batch_idx * H * (L + K) * D + head_idx * (L + K) * D;
    
    // Copy base keys and values
    for (int i = tid; i < base_size; i += blockDim.x) {
        int src_idx = base_offset + i;
        int dst_idx = output_base_offset + i;
        output_k[dst_idx] = base_k[src_idx];
        output_v[dst_idx] = base_v[src_idx];
    }
    
    __syncthreads();
    
    // Append accepted draft positions
    int draft_offset = batch_idx * H * K * D + head_idx * K * D;
    int output_draft_offset = output_base_offset + L * D;
    
    int accepted_count = 0;
    for (int k = 0; k < K && accepted_count < num_accepted; k++) {
        int mask_idx = batch_idx * K + k;
        if (accepted_mask[mask_idx]) {
            int src_offset = draft_offset + k * D;
            int dst_offset = output_draft_offset + accepted_count * D;
            
            // Copy this position's key and value
            for (int d = tid; d < D; d += blockDim.x) {
                output_k[dst_offset + d] = draft_k[src_offset + d];
                output_v[dst_offset + d] = draft_v[src_offset + d];
            }
            accepted_count++;
        }
    }
}

// Wrapper function
std::vector<torch::Tensor> kv_append_cuda(
    torch::Tensor base_k,
    torch::Tensor base_v,
    torch::Tensor draft_k,
    torch::Tensor draft_v,
    torch::Tensor accepted_mask,
    torch::Tensor accept_len,
    int offset
) {
    // Input validation
    TORCH_CHECK(base_k.dim() == 4, "base_k must be 4D tensor [B][H][L][D]");
    TORCH_CHECK(base_v.dim() == 4, "base_v must be 4D tensor [B][H][L][D]");
    TORCH_CHECK(draft_k.dim() == 4, "draft_k must be 4D tensor [B][H][K][D]");
    TORCH_CHECK(draft_v.dim() == 4, "draft_v must be 4D tensor [B][H][K][D]");
    TORCH_CHECK(accepted_mask.dim() == 2, "accepted_mask must be 2D tensor [B][K]");
    TORCH_CHECK(accept_len.dim() == 1, "accept_len must be 1D tensor [B]");
    
    TORCH_CHECK(base_k.device().is_cuda(), "base_k must be on CUDA");
    TORCH_CHECK(base_v.device().is_cuda(), "base_v must be on CUDA");
    TORCH_CHECK(draft_k.device().is_cuda(), "draft_k must be on CUDA");
    TORCH_CHECK(draft_v.device().is_cuda(), "draft_v must be on CUDA");
    
    int B = base_k.size(0);
    int H = base_k.size(1);
    int L = base_k.size(2);
    int K = draft_k.size(2);
    int D = base_k.size(3);
    
    // Create output tensors
    auto output_k = torch::zeros({B, H, L + K, D}, base_k.options());
    auto output_v = torch::zeros({B, H, L + K, D}, base_v.options());
    
    // Launch kernel
    dim3 block(256);
    dim3 grid(H, B);
    
    if (base_k.dtype() == torch::kFloat32) {
        kv_append_kernel<float><<<grid, block>>>(
            base_k.data_ptr<float>(),
            base_v.data_ptr<float>(),
            draft_k.data_ptr<float>(),
            draft_v.data_ptr<float>(),
            output_k.data_ptr<float>(),
            output_v.data_ptr<float>(),
            accepted_mask.data_ptr<uint8_t>(),
            accept_len.data_ptr<int32_t>(),
            B, H, L, K, D, offset
        );
    } else if (base_k.dtype() == torch::kFloat16) {
        kv_append_kernel<half><<<grid, block>>>(
            base_k.data_ptr<half>(),
            base_v.data_ptr<half>(),
            draft_k.data_ptr<half>(),
            draft_v.data_ptr<half>(),
            output_k.data_ptr<half>(),
            output_v.data_ptr<half>(),
            accepted_mask.data_ptr<uint8_t>(),
            accept_len.data_ptr<int32_t>(),
            B, H, L, K, D, offset
        );
    } else {
        throw std::runtime_error("Unsupported dtype for KV cache append");
    }
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(err)));
    }
    
    return {output_k, output_v};
}

// Python binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kv_append", &kv_append_cuda, "Append KV cache (CUDA)");
}
