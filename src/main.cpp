#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cmath>
#include <string>
#include "hybrid_engine.h"

int main(int argc, char* argv[]) {
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " <centroids.bin> <vectors.bin> <query_vectors.bin> <k> [num_queries] [top_n_clusters]\n";
        return 1;
    }

    std::string centroids_path = argv[1];
    std::string vectors_path = argv[2];
    std::string query_path = argv[3];
    int k = std::stoi(argv[4]);
    int num_queries_limit = (argc >= 6) ? std::stoi(argv[5]) : -1;
    int top_n_clusters = (argc >= 7) ? std::stoi(argv[6]) : 8;

    // Initialize Hybrid Engine
    HybridEngine engine;
    if (!engine.init(centroids_path, vectors_path)) {
        std::cerr << "Failed to initialize Hybrid Engine\n";
        return 1;
    }

    // Read Query Vectors
    std::ifstream query_file(query_path, std::ios::binary);
    if (!query_file.is_open()) {
        std::cerr << "Failed to open query file: " << query_path << "\n";
        return 1;
    }

    uint32_t query_header[2];
    query_file.read((char*)query_header, 8);
    uint32_t num_queries = query_header[0];
    uint32_t query_dim = query_header[1];

    if (query_dim != engine.get_dimension()) {
        std::cerr << "Query dimension (" << query_dim << ") mismatch with engine dimension (" 
                  << engine.get_dimension() << ")\n";
        return 1;
    }

    std::vector<float> query_data(num_queries * query_dim);
    query_file.read((char*)query_data.data(), num_queries * query_dim * sizeof(float));
    query_file.close();

    if (num_queries_limit > 0 && (uint32_t)num_queries_limit < num_queries) {
        num_queries = num_queries_limit;
    }

    std::cout << "[Hybrid Engine] Running " << num_queries << " queries (K=" << k 
              << ", top_n_clusters=" << top_n_clusters << ")...\n";

    // Warm-up query (to trigger CUDA initialization overhead outside our timing loop)
    engine.search(query_data.data(), k, top_n_clusters);

    double total_time_ms = 0.0;

    for (uint32_t q = 0; q < num_queries; ++q) {
        const float* query = &query_data[q * query_dim];

        auto start_time = std::chrono::high_resolution_clock::now();

        std::vector<DistancePair> results = engine.search(query, k, top_n_clusters);

        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end_time - start_time;
        total_time_ms += elapsed.count();

        // Print results for the first few queries
        if (q < 5) {
            std::cout << "Query " << q << " Top " << k << " Results:\n";
            for (int r = 0; r < k; ++r) {
                std::cout << "  Rank " << r 
                          << ": ID=" << results[r].id 
                          << ", Distance=" << std::sqrt(results[r].distance) << "\n";
            }
        }
    }

    std::cout << "[Hybrid Engine] Mean search latency: " << (total_time_ms / num_queries) << " ms\n";

    return 0;
}
