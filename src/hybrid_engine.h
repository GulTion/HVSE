#ifndef HYBRID_ENGINE_H
#define HYBRID_ENGINE_H

#include <string>
#include <vector>
#include "common.h"

class HybridEngine {
public:
    HybridEngine();
    ~HybridEngine();

    // Initialize the engine by mapping the centroids and vector database
    bool init(const std::string& centroids_path, const std::string& vectors_path);

    // Search for the top K closest vectors for a given query vector
    std::vector<DistancePair> search(const float* query_vector, int k, int top_n_clusters = 8);

    // Get database info
    uint32_t get_num_vectors() const { return num_vectors; }
    uint32_t get_dimension() const { return dimension; }
    uint32_t get_num_clusters() const { return num_clusters; }

private:
    // Memory mapping variables for centroids
    int centroids_fd;
    void* centroids_map;
    size_t centroids_map_size;
    float* centroids_data;

    // Memory mapping variables for vectors index
    int vectors_fd;
    void* vectors_map;
    size_t vectors_map_size;
    
    // Pointers into vectors mapped file
    uint32_t num_clusters;
    uint32_t dimension;
    uint32_t num_vectors;
    uint32_t* cluster_sizes;
    uint32_t* cluster_offsets;
    uint32_t* vector_ids;
    float* vector_data;

    // Preallocated CUDA resources to eliminate allocation overhead during queries
    uint32_t max_candidates;
    uint32_t max_k;
    
    // Device pointers
    float* d_query;
    float* d_candidates;
    DistancePair* d_distances;
    DistancePair* d_results;

    // Pinned host pointers
    float* h_pinned_query;
    float* h_pinned_candidates;
    uint32_t* h_pinned_candidate_ids;
    DistancePair* h_pinned_results;

    bool cuda_initialized;

    // Internal function to clean up mapped memory
    void cleanup_mmap();
};

#endif // HYBRID_ENGINE_H
