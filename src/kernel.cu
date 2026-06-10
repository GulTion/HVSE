#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include "common.h"

// CUDA Stream for asynchronous execution
static cudaStream_t stream;
static uint32_t* d_candidate_ids = nullptr;

// Inline device helpers
__device__ void insert_into_topk(DistancePair* topk, DistancePair pair, uint32_t k) {
    if (pair.distance >= topk[k - 1].distance) {
        return;
    }
    int i = k - 2;
    while (i >= 0 && topk[i].distance > pair.distance) {
        topk[i + 1] = topk[i];
        i--;
    }
    topk[i + 1] = pair;
}

__device__ void merge_topk(DistancePair* a, const DistancePair* b, uint32_t k) {
    DistancePair temp[256]; // supports up to K = 256
    uint32_t ia = 0, ib = 0, ic = 0;
    while (ic < k) {
        if (ia < k && (ib >= k || a[ia].distance <= b[ib].distance)) {
            temp[ic++] = a[ia++];
        } else {
            temp[ic++] = b[ib++];
        }
    }
    for (uint32_t i = 0; i < k; ++i) {
        a[i] = temp[i];
    }
}

// Kernel 1: Calculate distances
__global__ void compute_distances_kernel(
    const float* d_query,
    const float* d_candidates,
    const uint32_t* d_candidate_ids,
    uint32_t num_candidates,
    uint32_t dimension,
    DistancePair* d_distances
) {
    uint32_t c = blockIdx.x;
    if (c >= num_candidates) return;

    const float* candidate_vec = d_candidates + c * dimension;
    uint32_t t = threadIdx.x;

    float sum = 0.0f;
    for (uint32_t d = t; d < dimension; d += blockDim.x) {
        float diff = d_query[d] - candidate_vec[d];
        sum += diff * diff;
    }

    // Shared memory for block-level reduction
    __shared__ float sdata[256];
    sdata[t] = sum;
    __syncthreads();

    // Reduction tree
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (t < s) {
            sdata[t] += sdata[t + s];
        }
        __syncthreads();
    }

    if (t == 0) {
        d_distances[c].distance = sdata[0];
        d_distances[c].id = d_candidate_ids[c];
    }
}

// Kernel 2: Find top-K using parallel reduction
__global__ void reduction_topk_kernel(
    const DistancePair* d_distances,
    uint32_t num_candidates,
    uint32_t k,
    DistancePair* d_results
) {
    extern __shared__ DistancePair shared_topk[];

    uint32_t tid = threadIdx.x;
    DistancePair* my_topk = shared_topk + tid * k;

    // Initialize local top-K
    for (uint32_t i = 0; i < k; ++i) {
        my_topk[i].distance = INFINITY;
        my_topk[i].id = 0xFFFFFFFFu;
    }

    // Sequentially process candidate distances
    for (uint32_t idx = tid; idx < num_candidates; idx += blockDim.x) {
        DistancePair pair = d_distances[idx];
        insert_into_topk(my_topk, pair, k);
    }
    __syncthreads();

    // Tree reduction
    for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            DistancePair* other_topk = shared_topk + (tid + s) * k;
            merge_topk(my_topk, other_topk, k);
        }
        __syncthreads();
    }

    // Write final output
    if (tid == 0) {
        for (uint32_t i = 0; i < k; ++i) {
            d_results[i] = my_topk[i];
        }
    }
}

// C-interface functions
extern "C" bool cuda_init_resources(
    uint32_t max_candidates,
    uint32_t dimension,
    uint32_t max_k,
    float** d_query,
    float** d_candidates,
    DistancePair** d_distances,
    DistancePair** d_results,
    float** h_pinned_query,
    float** h_pinned_candidates,
    uint32_t** h_pinned_candidate_ids,
    DistancePair** h_pinned_results
) {
    cudaError_t err;

    // Create stream
    err = cudaStreamCreate(&stream);
    if (err != cudaSuccess) {
        std::cerr << "cudaStreamCreate failed: " << cudaGetErrorString(err) << "\n";
        return false;
    }

    // Allocate Device Memory
    err = cudaMalloc(d_query, dimension * sizeof(float));
    if (err != cudaSuccess) return false;

    err = cudaMalloc(d_candidates, max_candidates * dimension * sizeof(float));
    if (err != cudaSuccess) return false;

    err = cudaMalloc(d_distances, max_candidates * sizeof(DistancePair));
    if (err != cudaSuccess) return false;

    err = cudaMalloc(d_results, max_k * sizeof(DistancePair));
    if (err != cudaSuccess) return false;

    err = cudaMalloc(&d_candidate_ids, max_candidates * sizeof(uint32_t));
    if (err != cudaSuccess) return false;

    // Allocate Pinned Host Memory
    err = cudaHostAlloc(h_pinned_query, dimension * sizeof(float), cudaHostAllocDefault);
    if (err != cudaSuccess) return false;

    err = cudaHostAlloc(h_pinned_candidates, max_candidates * dimension * sizeof(float), cudaHostAllocDefault);
    if (err != cudaSuccess) return false;

    err = cudaHostAlloc(h_pinned_candidate_ids, max_candidates * sizeof(uint32_t), cudaHostAllocDefault);
    if (err != cudaSuccess) return false;

    err = cudaHostAlloc(h_pinned_results, max_k * sizeof(DistancePair), cudaHostAllocDefault);
    if (err != cudaSuccess) return false;

    return true;
}

extern "C" void cuda_free_resources(
    float* d_query,
    float* d_candidates,
    DistancePair* d_distances,
    DistancePair* d_results,
    float* h_pinned_query,
    float* h_pinned_candidates,
    uint32_t* h_pinned_candidate_ids,
    DistancePair* h_pinned_results
) {
    if (d_query) cudaFree(d_query);
    if (d_candidates) cudaFree(d_candidates);
    if (d_distances) cudaFree(d_distances);
    if (d_results) cudaFree(d_results);
    if (d_candidate_ids) cudaFree(d_candidate_ids);

    if (h_pinned_query) cudaFreeHost(h_pinned_query);
    if (h_pinned_candidates) cudaFreeHost(h_pinned_candidates);
    if (h_pinned_candidate_ids) cudaFreeHost(h_pinned_candidate_ids);
    if (h_pinned_results) cudaFreeHost(h_pinned_results);

    if (stream) cudaStreamDestroy(stream);
}

extern "C" void cuda_search_execution(
    const float* h_pinned_query,
    const float* h_pinned_candidates,
    const uint32_t* h_pinned_candidate_ids,
    uint32_t num_candidates,
    uint32_t dimension,
    uint32_t k,
    DistancePair* h_pinned_results,
    float* d_query,
    float* d_candidates,
    DistancePair* d_distances,
    DistancePair* d_results
) {
    // 1. Async PCIe Copy Host -> Device
    cudaMemcpyAsync(d_query, h_pinned_query, dimension * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_candidates, h_pinned_candidates, num_candidates * dimension * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_candidate_ids, h_pinned_candidate_ids, num_candidates * sizeof(uint32_t), cudaMemcpyHostToDevice, stream);

    // 2. Launch distance calculation kernel (1 block per candidate, 256 threads per block)
    compute_distances_kernel<<<num_candidates, 256, 0, stream>>>(
        d_query, d_candidates, d_candidate_ids, num_candidates, dimension, d_distances
    );

    // 3. Launch dynamic shared memory reduction kernel (1 block, 256 threads)
    // Shared memory size = block_size * k * sizeof(DistancePair)
    uint32_t shared_mem_size = 256 * k * sizeof(DistancePair);
    reduction_topk_kernel<<<1, 256, shared_mem_size, stream>>>(
        d_distances, num_candidates, k, d_results
    );

    // 4. Async copy results Device -> Host
    cudaMemcpyAsync(h_pinned_results, d_results, k * sizeof(DistancePair), cudaMemcpyDeviceToHost, stream);

    // 5. Sync stream to ensure everything is done before returning
    cudaStreamSynchronize(stream);
}
