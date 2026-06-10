#include "hybrid_engine.h"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <immintrin.h>

// Forward declarations of CUDA helper functions
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
);

extern "C" void cuda_free_resources(
    float* d_query,
    float* d_candidates,
    DistancePair* d_distances,
    DistancePair* d_results,
    float* h_pinned_query,
    float* h_pinned_candidates,
    uint32_t* h_pinned_candidate_ids,
    DistancePair* h_pinned_results
);

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
);

HybridEngine::HybridEngine()
    : centroids_fd(-1), centroids_map(MAP_FAILED), centroids_map_size(0), centroids_data(nullptr),
      vectors_fd(-1), vectors_map(MAP_FAILED), vectors_map_size(0),
      num_clusters(0), dimension(0), num_vectors(0),
      cluster_sizes(nullptr), cluster_offsets(nullptr), vector_ids(nullptr), vector_data(nullptr),
      max_candidates(500000), max_k(256),
      d_query(nullptr), d_candidates(nullptr), d_distances(nullptr), d_results(nullptr),
      h_pinned_query(nullptr), h_pinned_candidates(nullptr), h_pinned_candidate_ids(nullptr), h_pinned_results(nullptr),
      cuda_initialized(false) {}

HybridEngine::~HybridEngine() {
    if (cuda_initialized) {
        cuda_free_resources(
            d_query, d_candidates, d_distances, d_results,
            h_pinned_query, h_pinned_candidates, h_pinned_candidate_ids, h_pinned_results
        );
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

bool HybridEngine::init(const std::string& centroids_path, const std::string& vectors_path) {
    cleanup_mmap();

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

    // 3. Initialize pre-allocated CUDA resources
    cuda_initialized = cuda_init_resources(
        max_candidates, dimension, max_k,
        &d_query, &d_candidates, &d_distances, &d_results,
        &h_pinned_query, &h_pinned_candidates, &h_pinned_candidate_ids, &h_pinned_results
    );

    if (!cuda_initialized) {
        std::cerr << "Failed to initialize CUDA resources\n";
        return false;
    }

    return true;
}

std::vector<DistancePair> HybridEngine::search(const float* query_vector, int k, int top_n_clusters) {
    if (!cuda_initialized) {
        std::cerr << "Engine not initialized\n";
        return {};
    }

    if (k > (int)max_k) {
        std::cerr << "Requested K (" << k << ") exceeds pre-allocated max K (" << max_k << "). Truncating.\n";
        k = max_k;
    }

    // 1. CPU Centroid Search (AVX2 Pre-filter)
    std::vector<DistancePair> centroid_distances(num_clusters);

    for (uint32_t c = 0; c < num_clusters; ++c) {
        const float* centroid = centroids_data + c * dimension;
        
        // AVX2 Distance calculation
        __m256 sum = _mm256_setzero_ps();
        for (uint32_t d = 0; d < dimension; d += 8) {
            __m256 q = _mm256_loadu_ps(query_vector + d);
            __m256 cent = _mm256_loadu_ps(centroid + d);
            __m256 diff = _mm256_sub_ps(q, cent);
            sum = _mm256_fmadd_ps(diff, diff, sum);
        }

        // Horizontal sum of the 8 float components of the AVX2 register
        // sum = (s0, s1, s2, s3, s4, s5, s6, s7)
        __m256 shuf = _mm256_shuffle_ps(sum, sum, _MM_SHUFFLE(1, 0, 3, 2));
        __m256 sums = _mm256_add_ps(sum, shuf);
        __m256 shuf2 = _mm256_shuffle_ps(sums, sums, _MM_SHUFFLE(0, 1, 0, 1));
        __m256 sums2 = _mm256_add_ps(sums, shuf2);
        
        float dist_sq = ((float*)&sums2)[0] + ((float*)&sums2)[4];
        centroid_distances[c] = {dist_sq, c};
    }

    // Sort to find the top_n_clusters closest clusters
    std::sort(centroid_distances.begin(), centroid_distances.end(), [](const DistancePair& a, const DistancePair& b) {
        return a.distance < b.distance;
    });

    // 2. Gather candidate vectors from mapped CPU memory
    uint32_t total_candidates = 0;
    for (int i = 0; i < top_n_clusters; ++i) {
        uint32_t c_id = centroid_distances[i].id;
        uint32_t c_size = cluster_sizes[c_id];
        uint32_t c_offset = cluster_offsets[c_id];

        if (total_candidates + c_size > max_candidates) {
            std::cerr << "Candidates count exceeds max_candidates limit (" << max_candidates 
                      << "). Current: " << total_candidates << ", adding: " << c_size 
                      << " (rank " << i << ", cluster ID " << c_id << "). Truncating.\n";
            break;
        }

        // Copy IDs to pinned host buffer
        std::memcpy(h_pinned_candidate_ids + total_candidates, vector_ids + c_offset, c_size * sizeof(uint32_t));

        // Copy Vector data to pinned host buffer
        std::memcpy(h_pinned_candidates + total_candidates * dimension, vector_data + c_offset * dimension, c_size * dimension * sizeof(float));

        total_candidates += c_size;
    }

    // Copy query vector to pinned host buffer
    std::memcpy(h_pinned_query, query_vector, dimension * sizeof(float));

    // 3. GPU CUDA Execution (Async PCIe transfer, Parallel Distances, Parallel Reduction)
    cuda_search_execution(
        h_pinned_query, h_pinned_candidates, h_pinned_candidate_ids,
        total_candidates, dimension, k, h_pinned_results,
        d_query, d_candidates, d_distances, d_results
    );

    // 4. Return results (which are already copied back to h_pinned_results)
    std::vector<DistancePair> results(k);
    std::memcpy(results.data(), h_pinned_results, k * sizeof(DistancePair));
    
    return results;
}
