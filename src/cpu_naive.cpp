#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <queue>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include "common.h"

// Max-heap comparator to keep the largest elements at the top
struct CompareDist {
    bool operator()(const DistancePair& a, const DistancePair& b) {
        return a.distance < b.distance; // Largest distance on top
    }
};

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <base_vectors.bin> <query_vectors.bin> <k> [num_queries]\n";
        return 1;
    }

    const char* base_path = argv[1];
    const char* query_path = argv[2];
    int k = std::stoi(argv[3]);
    int num_queries_limit = (argc >= 5) ? std::stoi(argv[4]) : -1;

    // Open and mmap base vectors
    int base_fd = open(base_path, O_RDONLY);
    if (base_fd < 0) {
        std::cerr << "Failed to open base vectors file: " << base_path << "\n";
        return 1;
    }

    struct stat base_sb;
    if (fstat(base_fd, &base_sb) < 0) {
        std::cerr << "Failed to get base vectors file size\n";
        close(base_fd);
        return 1;
    }

    void* base_map = mmap(NULL, base_sb.st_size, PROT_READ, MAP_SHARED, base_fd, 0);
    if (base_map == MAP_FAILED) {
        std::cerr << "Failed to mmap base vectors file\n";
        close(base_fd);
        return 1;
    }

    // Read base header
    uint32_t* base_header = (uint32_t*)base_map;
    uint32_t num_vectors = base_header[0];
    uint32_t dimension = base_header[1];
    float* base_data = (float*)((char*)base_map + 8);

    std::cout << "[Naive CPU] Loaded base vectors: " << num_vectors << " vectors, dim " << dimension << "\n";

    // Open and read query vectors (small, so we can just read into memory)
    std::ifstream query_file(query_path, std::ios::binary);
    if (!query_file.is_open()) {
        std::cerr << "Failed to open query vectors file: " << query_path << "\n";
        munmap(base_map, base_sb.st_size);
        close(base_fd);
        return 1;
    }

    uint32_t query_header[2];
    query_file.read((char*)query_header, 8);
    uint32_t num_queries = query_header[0];
    uint32_t query_dim = query_header[1];

    if (query_dim != dimension) {
        std::cerr << "Query dimension (" << query_dim << ") does not match base dimension (" << dimension << ")\n";
        munmap(base_map, base_sb.st_size);
        close(base_fd);
        return 1;
    }

    std::vector<float> query_data(num_queries * dimension);
    query_file.read((char*)query_data.data(), num_queries * dimension * sizeof(float));
    query_file.close();

    if (num_queries_limit > 0 && (uint32_t)num_queries_limit < num_queries) {
        num_queries = num_queries_limit;
    }

    std::cout << "[Naive CPU] Running " << num_queries << " queries (K=" << k << ")...\n";

    double total_time_ms = 0.0;

    for (uint32_t q = 0; q < num_queries; ++q) {
        const float* query = &query_data[q * dimension];
        
        auto start_time = std::chrono::high_resolution_clock::now();

        // Max-heap of size K
        std::priority_queue<DistancePair, std::vector<DistancePair>, CompareDist> pq;

        for (uint32_t i = 0; i < num_vectors; ++i) {
            const float* vec = &base_data[i * dimension];
            float dist_sq = 0.0f;
            
            for (uint32_t d = 0; d < dimension; ++d) {
                float diff = query[d] - vec[d];
                dist_sq += diff * diff;
            }

            DistancePair pair = {dist_sq, i};

            if (pq.size() < (size_t)k) {
                pq.push(pair);
            } else if (dist_sq < pq.top().distance) {
                pq.pop();
                pq.push(pair);
            }
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end_time - start_time;
        total_time_ms += elapsed.count();

        // Print results for the first few queries
        if (q < 5) {
            std::cout << "Query " << q << " Top " << k << " Results:\n";
            std::vector<DistancePair> results;
            while (!pq.empty()) {
                results.push_back(pq.top());
                pq.pop();
            }
            // Results are in descending order from heap, print in ascending order
            for (int r = results.size() - 1; r >= 0; --r) {
                std::cout << "  Rank " << (results.size() - 1 - r) 
                          << ": ID=" << results[r].id 
                          << ", Distance=" << std::sqrt(results[r].distance) << "\n";
            }
        }
    }

    std::cout << "[Naive CPU] Mean search latency: " << (total_time_ms / num_queries) << " ms\n";

    munmap(base_map, base_sb.st_size);
    close(base_fd);
    return 0;
}
