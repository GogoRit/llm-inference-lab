#include <torch/extension.h>
#include <cuda_runtime.h>

// CUDA kernel for in-place KV cache update
// Writes new key/value tokens directly into pre-allocated buffer at specified position
// Optimized for coalesced loads/stores with read-only cache hints

// IN-PLACE KV cache update kernel
// OPTIMIZATIONS APPLIED:
// 1. Improved grid configuration for better GPU occupancy
// 2. Coalesced memory access patterns
// 3. Read-only cache hints (__ldg) for input data
// 4. Direct in-place writes (no intermediate buffers)
template<typename T>
__global__ void kv_append_inplace_kernel(
    T* __restrict__ cache_k,                // [B][H][Max_L][D] - pre-allocated key cache buffer
    T* __restrict__ cache_v,                // [B][H][Max_L][D] - pre-allocated value cache buffer
    const T* __restrict__ new_k,            // [B][H][New_L][D] - new key tokens to write
    const T* __restrict__ new_v,            // [B][H][New_L][D] - new value tokens to write
    int B, int H, int Max_L, int New_L, int D,
    int start_pos                           // Position in sequence dimension to start writing
) {
    // OPTIMIZATION: Use 1D grid for better occupancy
    // Calculate batch and head indices from linear block index
    int linear_idx = blockIdx.x;
    int batch_idx = linear_idx / H;
    int head_idx = linear_idx % H;
    
    if (batch_idx >= B || head_idx >= H) return;
    
    int tid = threadIdx.x;
    const int block_size = blockDim.x;
    
    // Calculate stride offsets for the pre-allocated buffer
    // Stride formula: batch_idx * stride_B + head_idx * stride_H + seq_pos * stride_L + dim_idx
    // Where stride_B = H * Max_L * D, stride_H = Max_L * D, stride_L = D
    const int stride_B = H * Max_L * D;  // Stride per batch
    const int stride_H = Max_L * D;       // Stride per head
    const int stride_L = D;               // Stride per sequence position
    
    // Base offsets for this (batch, head) pair
    const int cache_base_offset = batch_idx * stride_B + head_idx * stride_H;
    const int new_base_offset = batch_idx * H * New_L * D + head_idx * New_L * D;
    
    // Write new tokens into cache buffer at start_pos
    // Each thread handles multiple dimensions for better coalescing
    const int total_elements = New_L * D;
    
    for (int i = tid; i < total_elements; i += block_size) {
        // Calculate sequence position and dimension within new tokens
        int seq_idx = i / D;
        int dim_idx = i % D;
        
        // Source: new_kv at [batch][head][seq_idx][dim_idx]
        int src_idx = new_base_offset + seq_idx * D + dim_idx;
        
        // Destination: cache at [batch][head][start_pos + seq_idx][dim_idx]
        int dst_seq_pos = start_pos + seq_idx;
        int dst_idx = cache_base_offset + dst_seq_pos * stride_L + dim_idx;
        
        // OPTIMIZATION: Use __ldg for read-only cache on input data
        // Direct write to cache (no intermediate buffer)
        cache_k[dst_idx] = __ldg(&new_k[src_idx]);
        cache_v[dst_idx] = __ldg(&new_v[src_idx]);
    }
}

// In-place KV cache update wrapper function
// Writes new tokens directly into pre-allocated cache buffer
void kv_append_inplace_cuda(
    torch::Tensor cache_k,      // [B][H][Max_L][D] - pre-allocated key cache buffer
    torch::Tensor cache_v,      // [B][H][Max_L][D] - pre-allocated value cache buffer
    torch::Tensor new_k,        // [B][H][New_L][D] - new key tokens to write
    torch::Tensor new_v,        // [B][H][New_L][D] - new value tokens to write
    int start_pos               // Position in sequence dimension to start writing
) {
    // Input validation
    TORCH_CHECK(cache_k.dim() == 4, "cache_k must be 4D tensor [B][H][Max_L][D]");
    TORCH_CHECK(cache_v.dim() == 4, "cache_v must be 4D tensor [B][H][Max_L][D]");
    TORCH_CHECK(new_k.dim() == 4, "new_k must be 4D tensor [B][H][New_L][D]");
    TORCH_CHECK(new_v.dim() == 4, "new_v must be 4D tensor [B][H][New_L][D]");
    
    TORCH_CHECK(cache_k.device().is_cuda(), "cache_k must be on CUDA");
    TORCH_CHECK(cache_v.device().is_cuda(), "cache_v must be on CUDA");
    TORCH_CHECK(new_k.device().is_cuda(), "new_k must be on CUDA");
    TORCH_CHECK(new_v.device().is_cuda(), "new_v must be on CUDA");
    
    // Extract dimensions
    int B = cache_k.size(0);
    int H = cache_k.size(1);
    int Max_L = cache_k.size(2);
    int D = cache_k.size(3);
    
    int New_L = new_k.size(2);
    
    // Validate dimensions match
    TORCH_CHECK(cache_v.size(0) == B, "cache_v batch size must match cache_k");
    TORCH_CHECK(cache_v.size(1) == H, "cache_v num_heads must match cache_k");
    TORCH_CHECK(cache_v.size(2) == Max_L, "cache_v max_seq_len must match cache_k");
    TORCH_CHECK(cache_v.size(3) == D, "cache_v head_dim must match cache_k");
    
    TORCH_CHECK(new_k.size(0) == B, "new_k batch size must match cache_k");
    TORCH_CHECK(new_k.size(1) == H, "new_k num_heads must match cache_k");
    TORCH_CHECK(new_k.size(3) == D, "new_k head_dim must match cache_k");
    
    TORCH_CHECK(new_v.size(0) == B, "new_v batch size must match cache_k");
    TORCH_CHECK(new_v.size(1) == H, "new_v num_heads must match cache_k");
    TORCH_CHECK(new_v.size(2) == New_L, "new_v seq_len must match new_k");
    TORCH_CHECK(new_v.size(3) == D, "new_v head_dim must match cache_k");
    
    // Validate start_pos and bounds
    TORCH_CHECK(start_pos >= 0, "start_pos must be non-negative");
    TORCH_CHECK(start_pos + New_L <= Max_L, 
                "start_pos + New_L (" + std::to_string(start_pos + New_L) + 
                ") exceeds Max_L (" + std::to_string(Max_L) + ")");
    
    // Validate dtypes match
    TORCH_CHECK(cache_k.dtype() == cache_v.dtype(), "cache_k and cache_v must have same dtype");
    TORCH_CHECK(new_k.dtype() == new_v.dtype(), "new_k and new_v must have same dtype");
    TORCH_CHECK(cache_k.dtype() == new_k.dtype(), "cache and new tensors must have same dtype");
    
    // Launch kernel with optimized grid configuration
    // OPTIMIZATION: Use 1D grid for better GPU occupancy
    const int threads_per_block = 256;
    dim3 block(threads_per_block);
    // 1D grid: One block per (batch, head) pair
    dim3 grid(B * H);
    
    // Dispatch based on dtype
    if (cache_k.dtype() == torch::kFloat32) {
        kv_append_inplace_kernel<float><<<grid, block>>>(
            cache_k.data_ptr<float>(),
            cache_v.data_ptr<float>(),
            new_k.data_ptr<float>(),
            new_v.data_ptr<float>(),
            B, H, Max_L, New_L, D,
            start_pos
        );
    } else if (cache_k.dtype() == torch::kFloat16) {
        kv_append_inplace_kernel<half><<<grid, block>>>(
            cache_k.data_ptr<half>(),
            cache_v.data_ptr<half>(),
            new_k.data_ptr<half>(),
            new_v.data_ptr<half>(),
            B, H, Max_L, New_L, D,
            start_pos
        );
    } else {
        throw std::runtime_error("Unsupported dtype for KV cache append: " + 
                                std::string(cache_k.dtype().name()));
    }
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA kernel launch failed: " + 
                                std::string(cudaGetErrorString(err)));
    }
}

// Python binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kv_append_inplace", &kv_append_inplace_cuda, 
          "In-place KV cache update (CUDA) - writes new tokens into pre-allocated buffer");
}
