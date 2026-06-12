#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <cstring>
#include "common.h"

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

// Kernel 1: Centroid Distance Matrix Computation
__global__ void compute_centroid_distances_kernel(
    const float* d_queries,
    const float* d_centroids,
    uint32_t num_queries,
    uint32_t num_centroids,
    uint32_t dimension,
    float* d_centroid_distances
) {
    uint32_t c = blockIdx.x; // Centroid index
    uint32_t q = blockIdx.y; // Query index
    if (c >= num_centroids || q >= num_queries) return;

    const float* query_vec = d_queries + q * dimension;
    const float* centroid_vec = d_centroids + c * dimension;
    uint32_t t = threadIdx.x;

    float sum = 0.0f;
    for (uint32_t d = t; d < dimension; d += blockDim.x) {
        float diff = query_vec[d] - centroid_vec[d];
        sum += diff * diff;
    }

    // Shared memory reduction
    __shared__ float sdata[256];
    sdata[t] = sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (t < s) {
            sdata[t] += sdata[t + s];
        }
        __syncthreads();
    }

    if (t == 0) {
        d_centroid_distances[q * num_centroids + c] = sdata[0];
    }
}

// Kernel 2: Top-N Centroid Search per Query
__global__ void find_top_clusters_kernel(
    const float* d_centroid_distances,
    uint32_t num_queries,
    uint32_t num_centroids,
    uint32_t top_n_clusters,
    DistancePair* d_top_clusters
) {
    uint32_t q = blockIdx.x; // Query index
    if (q >= num_queries) return;

    uint32_t tid = threadIdx.x;
    const float* my_distances = d_centroid_distances + q * num_centroids;

    extern __shared__ DistancePair shared_topn[];
    DistancePair* my_topn = shared_topn + tid * top_n_clusters;

    // Initialize local top-N
    for (uint32_t i = 0; i < top_n_clusters; ++i) {
        my_topn[i].distance = INFINITY;
        my_topn[i].id = 0xFFFFFFFFu;
    }

    // Sequentially process centroid distances
    for (uint32_t c = tid; c < num_centroids; c += blockDim.x) {
        DistancePair pair = {my_distances[c], c};
        insert_into_topk(my_topn, pair, top_n_clusters);
    }
    __syncthreads();

    // Tree reduction
    for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            DistancePair* other_topn = shared_topn + (tid + s) * top_n_clusters;
            merge_topk(my_topn, other_topn, top_n_clusters);
        }
        __syncthreads();
    }

    // Write final output
    if (tid == 0) {
        for (uint32_t i = 0; i < top_n_clusters; ++i) {
            d_top_clusters[q * top_n_clusters + i] = my_topn[i];
        }
    }
}

// Kernel 3: Calculate candidate vector distances
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

// Kernel 4: Find top-K candidate vectors using parallel reduction
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
    uint32_t num_streams,
    uint32_t max_candidates,
    uint32_t dimension,
    uint32_t max_k,
    const float* centroids_host,
    uint32_t num_centroids,
    float** d_centroids,
    StreamResources* streams_res
) {
    cudaError_t err;

    // Allocate and upload centroids (common to all queries)
    err = cudaMalloc(d_centroids, num_centroids * dimension * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc d_centroids failed: " << cudaGetErrorString(err) << "\n";
        return false;
    }
    err = cudaMemcpy(*d_centroids, centroids_host, num_centroids * dimension * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy centroids failed: " << cudaGetErrorString(err) << "\n";
        return false;
    }

    // Allocate resources for each stream
    for (uint32_t s = 0; s < num_streams; ++s) {
        StreamResources& res = streams_res[s];
        
        // Zero-initialize pointers so we can safely clean up if some fail
        std::memset(&res, 0, sizeof(StreamResources));

        // Create stream
        err = cudaStreamCreate(&res.stream);
        if (err != cudaSuccess) {
            std::cerr << "cudaStreamCreate failed at stream " << s << ": " << cudaGetErrorString(err) << "\n";
            return false;
        }

        // Allocate Device memory
        err = cudaMalloc(&res.d_query, dimension * sizeof(float));
        if (err != cudaSuccess) {
            std::cerr << "cudaMalloc d_query failed at stream " << s << ": " << cudaGetErrorString(err) << "\n";
            return false;
        }

        err = cudaMalloc(&res.d_candidates, max_candidates * dimension * sizeof(float));
        if (err != cudaSuccess) {
            std::cerr << "cudaMalloc d_candidates failed at stream " << s << ": " << cudaGetErrorString(err) << "\n";
            return false;
        }

        err = cudaMalloc(&res.d_distances, max_candidates * sizeof(DistancePair));
        if (err != cudaSuccess) {
            std::cerr << "cudaMalloc d_distances failed at stream " << s << ": " << cudaGetErrorString(err) << "\n";
            return false;
        }

        err = cudaMalloc(&res.d_results, max_k * sizeof(DistancePair));
        if (err != cudaSuccess) {
            std::cerr << "cudaMalloc d_results failed at stream " << s << ": " << cudaGetErrorString(err) << "\n";
            return false;
        }

        err = cudaMalloc(&res.d_candidate_ids, max_candidates * sizeof(uint32_t));
        if (err != cudaSuccess) {
            std::cerr << "cudaMalloc d_candidate_ids failed at stream " << s << ": " << cudaGetErrorString(err) << "\n";
            return false;
        }

        // Allocate Pinned Host memory
        err = cudaHostAlloc(&res.h_pinned_query, dimension * sizeof(float), cudaHostAllocDefault);
        if (err != cudaSuccess) {
            std::cerr << "cudaHostAlloc h_pinned_query failed at stream " << s << ": " << cudaGetErrorString(err) << "\n";
            return false;
        }

        err = cudaHostAlloc(&res.h_pinned_candidates, max_candidates * dimension * sizeof(float), cudaHostAllocDefault);
        if (err != cudaSuccess) {
            std::cerr << "cudaHostAlloc h_pinned_candidates failed at stream " << s << ": " << cudaGetErrorString(err) << "\n";
            return false;
        }

        err = cudaHostAlloc(&res.h_pinned_candidate_ids, max_candidates * sizeof(uint32_t), cudaHostAllocDefault);
        if (err != cudaSuccess) {
            std::cerr << "cudaHostAlloc h_pinned_candidate_ids failed at stream " << s << ": " << cudaGetErrorString(err) << "\n";
            return false;
        }

        err = cudaHostAlloc(&res.h_pinned_results, max_k * sizeof(DistancePair), cudaHostAllocDefault);
        if (err != cudaSuccess) {
            std::cerr << "cudaHostAlloc h_pinned_results failed at stream " << s << ": " << cudaGetErrorString(err) << "\n";
            return false;
        }
    }

    return true;
}

extern "C" void cuda_free_resources(
    uint32_t num_streams,
    float* d_centroids,
    StreamResources* streams_res
) {
    if (d_centroids) {
        cudaFree(d_centroids);
    }
    for (uint32_t s = 0; s < num_streams; ++s) {
        StreamResources& res = streams_res[s];
        if (res.d_query) cudaFree(res.d_query);
        if (res.d_candidates) cudaFree(res.d_candidates);
        if (res.d_distances) cudaFree(res.d_distances);
        if (res.d_results) cudaFree(res.d_results);
        if (res.d_candidate_ids) cudaFree(res.d_candidate_ids);

        if (res.h_pinned_query) cudaFreeHost(res.h_pinned_query);
        if (res.h_pinned_candidates) cudaFreeHost(res.h_pinned_candidates);
        if (res.h_pinned_candidate_ids) cudaFreeHost(res.h_pinned_candidate_ids);
        if (res.h_pinned_results) cudaFreeHost(res.h_pinned_results);

        if (res.stream) cudaStreamDestroy(res.stream);
    }
}

extern "C" void cuda_centroid_search(
    const float* queries_host,
    uint32_t num_queries,
    uint32_t dimension,
    const float* d_centroids,
    uint32_t num_centroids,
    uint32_t top_n_clusters,
    DistancePair* top_clusters_host
) {
    cudaError_t err;
    float* d_queries = nullptr;
    float* d_centroid_distances = nullptr;
    DistancePair* d_top_clusters = nullptr;

    err = cudaMalloc(&d_queries, num_queries * dimension * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc d_queries failed: " << cudaGetErrorString(err) << "\n";
        return;
    }
    
    err = cudaMalloc(&d_centroid_distances, num_queries * num_centroids * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc d_centroid_distances failed: " << cudaGetErrorString(err) << "\n";
        cudaFree(d_queries);
        return;
    }

    err = cudaMalloc(&d_top_clusters, num_queries * top_n_clusters * sizeof(DistancePair));
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc d_top_clusters failed: " << cudaGetErrorString(err) << "\n";
        cudaFree(d_queries);
        cudaFree(d_centroid_distances);
        return;
    }

    // Copy queries to device
    err = cudaMemcpy(d_queries, queries_host, num_queries * dimension * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy queries_host failed: " << cudaGetErrorString(err) << "\n";
        cudaFree(d_queries);
        cudaFree(d_centroid_distances);
        cudaFree(d_top_clusters);
        return;
    }

    // Launch distance matrix kernel
    dim3 grid_dist(num_centroids, num_queries);
    compute_centroid_distances_kernel<<<grid_dist, 256>>>(
        d_queries, d_centroids, num_queries, num_centroids, dimension, d_centroid_distances
    );
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "compute_centroid_distances_kernel launch failed: " << cudaGetErrorString(err) << "\n";
        cudaFree(d_queries);
        cudaFree(d_centroid_distances);
        cudaFree(d_top_clusters);
        return;
    }

    // Launch top-N cluster search kernel (using block size 32 to fit shared memory limit)
    uint32_t block_size = 32;
    uint32_t shared_mem_size = block_size * top_n_clusters * sizeof(DistancePair);
    find_top_clusters_kernel<<<num_queries, block_size, shared_mem_size>>>(
        d_centroid_distances, num_queries, num_centroids, top_n_clusters, d_top_clusters
    );

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "find_top_clusters_kernel launch failed: " << cudaGetErrorString(err) << "\n";
        cudaFree(d_queries);
        cudaFree(d_centroid_distances);
        cudaFree(d_top_clusters);
        return;
    }

    // Copy top clusters back to host
    err = cudaMemcpy(top_clusters_host, d_top_clusters, num_queries * top_n_clusters * sizeof(DistancePair), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy d_top_clusters back failed: " << cudaGetErrorString(err) << "\n";
    }

    // Free resources
    cudaFree(d_queries);
    cudaFree(d_centroid_distances);
    cudaFree(d_top_clusters);
}

extern "C" void cuda_search_execution_async(
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
    DistancePair* d_results,
    uint32_t* d_candidate_ids,
    cudaStream_t stream
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
    uint32_t shared_mem_size = 256 * k * sizeof(DistancePair);
    reduction_topk_kernel<<<1, 256, shared_mem_size, stream>>>(
        d_distances, num_candidates, k, d_results
    );

    // 4. Async copy results Device -> Host
    cudaMemcpyAsync(h_pinned_results, d_results, k * sizeof(DistancePair), cudaMemcpyDeviceToHost, stream);
}
