#ifndef COMMON_H
#define COMMON_H

#include <stdint.h>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

struct VectorsHeader {
    uint32_t num_clusters;
    uint32_t dimension;
    uint32_t num_vectors;
};

struct CentroidsHeader {
    uint32_t num_clusters;
    uint32_t dimension;
};

struct DistancePair {
    float distance;
    uint32_t id;
};

struct StreamResources {
    cudaStream_t stream;
    float* d_query;
    float* d_candidates;
    DistancePair* d_distances;
    DistancePair* d_results;
    uint32_t* d_candidate_ids;
    float* h_pinned_query;
    float* h_pinned_candidates;
    uint32_t* h_pinned_candidate_ids;
    DistancePair* h_pinned_results;
};

#ifdef __cplusplus
}
#endif

#endif // COMMON_H
