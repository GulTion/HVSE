#ifndef COMMON_H
#define COMMON_H

#include <stdint.h>

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

#ifdef __cplusplus
}
#endif

#endif // COMMON_H
