#include "hybrid_engine.h"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

// Forward declarations of CUDA helper functions
extern "C" bool cuda_init_resources(
    uint32_t num_streams,
    uint32_t max_candidates,
    uint32_t dimension,
    uint32_t max_k,
    const float* centroids_host,
    uint32_t num_centroids,
    float** d_centroids,
    StreamResources* streams_res
);

extern "C" void cuda_free_resources(
    uint32_t num_streams,
    float* d_centroids,
    StreamResources* streams_res
);

extern "C" void cuda_centroid_search(
    const float* queries_host,
    uint32_t num_queries,
    uint32_t dimension,
    const float* d_centroids,
    uint32_t num_centroids,
    uint32_t top_n_clusters,
    DistancePair* top_clusters_host
);

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
);

HybridEngine::HybridEngine()
    : centroids_fd(-1), centroids_map(MAP_FAILED), centroids_map_size(0), centroids_data(nullptr),
      vectors_fd(-1), vectors_map(MAP_FAILED), vectors_map_size(0),
      num_clusters(0), dimension(0), num_vectors(0),
      cluster_sizes(nullptr), cluster_offsets(nullptr), vector_ids(nullptr), vector_data(nullptr),
      num_streams(8), max_candidates(32768), max_k(256),
      d_centroids(nullptr),
      cuda_initialized(false) {}

HybridEngine::~HybridEngine() {
    if (cuda_initialized) {
        cuda_free_resources(num_streams, d_centroids, streams_res.data());
    }
    cleanup_mmap();
}

void HybridEngine::cleanup_mmap() {
    if (centroids_map != MAP_FAILED) {
        munmap(centroids_map, centroids_map_size);
        centroids_map = MAP_FAILED;
    }
    if (centroids_fd >= 0) {
        close(centroids_fd);
        centroids_fd = -1;
    }

    if (vectors_map != MAP_FAILED) {
        munmap(vectors_map, vectors_map_size);
        vectors_map = MAP_FAILED;
    }
    if (vectors_fd >= 0) {
        close(vectors_fd);
        vectors_fd = -1;
    }
}

bool HybridEngine::init(const std::string& centroids_path, const std::string& vectors_path, int input_num_streams) {
    cleanup_mmap();
    num_streams = input_num_streams;

    // 1. Map centroids
    centroids_fd = open(centroids_path.c_str(), O_RDONLY);
    if (centroids_fd < 0) {
        std::cerr << "Failed to open centroids file: " << centroids_path << "\n";
        return false;
    }

    struct stat cent_sb;
    if (fstat(centroids_fd, &cent_sb) < 0) {
        std::cerr << "Failed to get centroids file stat\n";
        return false;
    }
    centroids_map_size = cent_sb.st_size;

    centroids_map = mmap(NULL, centroids_map_size, PROT_READ, MAP_SHARED, centroids_fd, 0);
    if (centroids_map == MAP_FAILED) {
        std::cerr << "Failed to mmap centroids file\n";
        return false;
    }

    CentroidsHeader* cent_hdr = (CentroidsHeader*)centroids_map;
    uint32_t cent_clusters = cent_hdr->num_clusters;
    uint32_t cent_dim = cent_hdr->dimension;
    centroids_data = (float*)((char*)centroids_map + sizeof(CentroidsHeader));

    // 2. Map vectors database
    vectors_fd = open(vectors_path.c_str(), O_RDONLY);
    if (vectors_fd < 0) {
        std::cerr << "Failed to open vectors database file: " << vectors_path << "\n";
        return false;
    }

    struct stat vec_sb;
    if (fstat(vectors_fd, &vec_sb) < 0) {
        std::cerr << "Failed to get vectors file stat\n";
        return false;
    }
    vectors_map_size = vec_sb.st_size;

    vectors_map = mmap(NULL, vectors_map_size, PROT_READ, MAP_SHARED, vectors_fd, 0);
    if (vectors_map == MAP_FAILED) {
        std::cerr << "Failed to mmap vectors file\n";
        return false;
    }

    VectorsHeader* vec_hdr = (VectorsHeader*)vectors_map;
    num_clusters = vec_hdr->num_clusters;
    dimension = vec_hdr->dimension;
    num_vectors = vec_hdr->num_vectors;

    if (num_clusters != cent_clusters || dimension != cent_dim) {
        std::cerr << "Dimension or cluster count mismatch between centroids and vectors file\n";
        return false;
    }

    // Set up pointers into vectors mapped file
    char* base_ptr = (char*)vectors_map + sizeof(VectorsHeader);
    cluster_sizes = (uint32_t*)base_ptr;
    cluster_offsets = (uint32_t*)(base_ptr + num_clusters * sizeof(uint32_t));
    vector_ids = (uint32_t*)(base_ptr + 2 * num_clusters * sizeof(uint32_t));
    vector_data = (float*)(base_ptr + 2 * num_clusters * sizeof(uint32_t) + num_vectors * sizeof(uint32_t));

    std::cout << "[Hybrid Engine] Mapped index: " << num_vectors << " vectors, " 
              << num_clusters << " clusters, dim " << dimension << "\n";

    // 3. Initialize pre-allocated CUDA resources for all streams
    streams_res.resize(num_streams);
    cuda_initialized = cuda_init_resources(
        num_streams, max_candidates, dimension, max_k,
        centroids_data, num_clusters, &d_centroids, streams_res.data()
    );

    if (!cuda_initialized) {
        std::cerr << "Failed to initialize CUDA resources\n";
        return false;
    }

    return true;
}

std::vector<DistancePair> HybridEngine::search(const float* query_vector, int k, int top_n_clusters) {
    auto batch_results = search_batch(query_vector, 1, k, top_n_clusters);
    return batch_results[0];
}

std::vector<std::vector<DistancePair>> HybridEngine::search_batch(
    const float* queries_vector,
    int num_queries,
    int k,
    int top_n_clusters
) {
    if (!cuda_initialized) {
        std::cerr << "Engine not initialized\n";
        return {};
    }

    if (k > (int)max_k) {
        std::cerr << "Requested K (" << k << ") exceeds pre-allocated max K (" << max_k << "). Truncating.\n";
        k = max_k;
    }

    // 1. GPU Centroid Search (Parallel Query, Parallel Centroid, Parallel Reduction)
    std::vector<DistancePair> h_top_clusters(num_queries * top_n_clusters);
    cuda_centroid_search(
        queries_vector, num_queries, dimension,
        d_centroids, num_clusters, top_n_clusters,
        h_top_clusters.data()
    );

    // 2. Pipelined Loop over all queries using B CUDA streams
    std::vector<std::vector<DistancePair>> batch_results(num_queries, std::vector<DistancePair>(k));

    for (int q = 0; q < num_queries; ++q) {
        int s = q % num_streams;

        // Synchronize stream s to finish its previous query (if any) before launching a new one
        if (q >= (int)num_streams) {
            int prev_q = q - num_streams;
            cudaStreamSynchronize(streams_res[s].stream);
            std::memcpy(batch_results[prev_q].data(), streams_res[s].h_pinned_results, k * sizeof(DistancePair));
        }

        // Gather candidates from mapped CPU memory
        uint32_t total_candidates = 0;
        const DistancePair* q_top_clusters = &h_top_clusters[q * top_n_clusters];

        for (int i = 0; i < top_n_clusters; ++i) {
            uint32_t c_id = q_top_clusters[i].id;
            if (c_id == 0xFFFFFFFFu) continue;

            uint32_t c_size = cluster_sizes[c_id];
            uint32_t c_offset = cluster_offsets[c_id];

            if (total_candidates + c_size > max_candidates) {
                // Truncate safely
                c_size = max_candidates - total_candidates;
            }

            if (c_size == 0) break;

            // Copy IDs and Vector data to stream's pinned host buffers
            std::memcpy(streams_res[s].h_pinned_candidate_ids + total_candidates, vector_ids + c_offset, c_size * sizeof(uint32_t));
            std::memcpy(streams_res[s].h_pinned_candidates + total_candidates * dimension, vector_data + c_offset * dimension, c_size * dimension * sizeof(float));

            total_candidates += c_size;
            if (total_candidates >= max_candidates) break;
        }

        // Copy query vector to stream's pinned host buffer
        std::memcpy(streams_res[s].h_pinned_query, queries_vector + q * dimension, dimension * sizeof(float));

        // Launch async copy & search kernels on stream s
        cuda_search_execution_async(
            streams_res[s].h_pinned_query,
            streams_res[s].h_pinned_candidates,
            streams_res[s].h_pinned_candidate_ids,
            total_candidates,
            dimension,
            k,
            streams_res[s].h_pinned_results,
            streams_res[s].d_query,
            streams_res[s].d_candidates,
            streams_res[s].d_distances,
            streams_res[s].d_results,
            streams_res[s].d_candidate_ids,
            streams_res[s].stream
        );
    }

    // 3. Final synchronization loop for remaining active streams
    for (int s = 0; s < (int)num_streams; ++s) {
        for (int q = num_queries - (int)num_streams; q < num_queries; ++q) {
            if (q >= 0 && (q % (int)num_streams) == s) {
                cudaStreamSynchronize(streams_res[s].stream);
                std::memcpy(batch_results[q].data(), streams_res[s].h_pinned_results, k * sizeof(DistancePair));
            }
        }
    }

    return batch_results;
}
