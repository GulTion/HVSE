# Hybrid Vector Search Engine

A high-performance Hybrid Vector Search Engine that combines CPU AVX2 SIMD pre-filtering with custom CUDA execution and dynamic shared-memory parallel reduction to search 1,000,000+ high-dimensional vectors. 

The system is optimized for VRAM-constrained hardware, utilizing CPU `mmap()` virtual memory mapping to store the database, and streaming only candidate vector subsets asynchronously over PCIe to the GPU. This allows searching massive datasets on low-end local GPUs (e.g., **GeForce MX350 with 2GB VRAM**) and easily scaling to 10M+ vectors on Google Colab (**Tesla T4 with 16GB VRAM**).

---

## 📂 Repository Structure

- **`data/`**: Root directory for data files (auto-generated, git-ignored).
- **`scripts/`**: Python utility and benchmarking scripts.
  - [generate_data.py](file:///media/gultion/Material/Projects/Hybrid Vector Search Engine/scripts/generate_data.py): Generates base and query synthetic datasets.
  - [build_index.py](file:///media/gultion/Material/Projects/Hybrid Vector Search Engine/scripts/build_index.py): Performs K-Means clustering and serializes the IVF database layout.
  - [benchmark.py](file:///media/gultion/Material/Projects/Hybrid Vector Search Engine/scripts/benchmark.py): Runs and measures latencies for CPU Naive, Hybrid, PyTorch GPU, and Faiss.
- **`src/`**: C++ and CUDA source code.
  - [common.h](file:///media/gultion/Material/Projects/Hybrid Vector Search Engine/src/common.h): File format headers and result structures.
  - [cpu_naive.cpp](file:///media/gultion/Material/Projects/Hybrid Vector Search Engine/src/cpu_naive.cpp): Exact sequential CPU search baseline.
  - [hybrid_engine.h](file:///media/gultion/Material/Projects/Hybrid Vector Search Engine/src/hybrid_engine.h): Host-side hybrid search class interface.
  - [hybrid_engine.cpp](file:///media/gultion/Material/Projects/Hybrid Vector Search Engine/src/hybrid_engine.cpp): Implementation of memory mapping and AVX2 centroid pre-filter.
  - [kernel.cu](file:///media/gultion/Material/Projects/Hybrid Vector Search Engine/src/kernel.cu): CUDA distance kernels and dynamic shared memory top-K reduction.
  - [main.cpp](file:///media/gultion/Material/Projects/Hybrid Vector Search Engine/src/main.cpp): CLI executable wrapper for query testing.
- [Makefile](file:///media/gultion/Material/Projects/Hybrid Vector Search Engine/Makefile): Compilation settings for Pascal (local) and Turing (Colab) GPU architectures.

---

## ⚙️ Environment Setup

### 1. Local Environment Setup (using `uv`)
Initialize a Python virtual environment and install the required dependencies (excluding PyTorch to save space):
```bash
# Create the virtual environment
uv venv

# Install scikit-learn and numpy
uv pip install numpy scikit-learn
```

### 2. Google Colab Environment Setup
When deploying to Colab, install the dependencies system-wide (including PyTorch and GPU-accelerated Faiss):
```bash
pip install numpy scikit-learn torch faiss-gpu
```

---

## 🔨 Compilation Instructions

The project uses a Makefile that detects compilation targets to match compute capabilities:
- **Local (Compute Capability 6.1 - Pascal, MX350)**
- **Google Colab (Compute Capability 7.5 - Turing, T4)**

### Compile locally:
```bash
make clean && make
```

### Compile on Google Colab:
```bash
make clean && make TARGET=colab
```

---

## 🏃 Run Instructions

### Step 1: Generate Data
Generate 1,000,000 base vectors (512-dim) and 100 query vectors:
```bash
.venv/bin/python3 scripts/generate_data.py
```
*(Or `python3 scripts/generate_data.py` on Colab)*

### Step 2: Build the IVF Index
Cluster the vectors into 1,024 clusters and group them contiguously:
```bash
.venv/bin/python3 scripts/build_index.py
```
*(Or `python3 scripts/build_index.py` on Colab)*

### Step 3: Run Searches Manually

#### Run Sequential CPU Search (Exact):
```bash
./cpu_naive data/base_vectors.bin data/query_vectors.bin <K> [num_queries]
```
Example (retrieve top 5 closest vectors for first 10 queries):
```bash
./cpu_naive data/base_vectors.bin data/query_vectors.bin 5 10
```

#### Run Hybrid Search (Approximate):
```bash
./hybrid_search data/centroids.bin data/vectors.bin data/query_vectors.bin <K> [num_queries] [top_n_clusters]
```
Example (retrieve top 5 closest vectors searching 8 closest clusters for the first 10 queries):
```bash
./hybrid_search data/centroids.bin data/vectors.bin data/query_vectors.bin 5 10 8
```

---

## 📊 Benchmarking Execution

To run the comparison suite and print accuracy/latency statistics:
```bash
.venv/bin/python3 scripts/benchmark.py
```
*(Or `python3 scripts/benchmark.py` on Colab)*

### Output Metrics
The benchmark script automatically runs the benchmarks and outputs:
- **Correctness and Recall Verification**: Computes the top-K overlap (Recall) between the approximate IVF Hybrid Engine and the exact CPU Naive search.
- **Performance Summary**: A markdown table showing the mean search latencies of all active engines (CPU Naive, PyTorch GPU, Faiss L2/IVF, and Hybrid Engine).

---

## 🎛️ Hyperparameters Tuning for Custom Benchmarks

To customize benchmarks, you can adjust the following parameters:

### 1. Scale of the Dataset
To change the number of vectors, query vectors, or vector dimension:
- Open [generate_data.py](file:///media/gultion/Material/Projects/Hybrid Vector Search Engine/scripts/generate_data.py).
- Edit these variables at the beginning of `generate_and_save_data()`:
  ```python
  num_base = 1000000   # Set to 10000000 for 10 million vectors (Colab T4 benchmark)
  num_query = 100      # Number of query vectors
  dim = 512            # Vector dimensions (must be a multiple of 8 for AVX2)
  ```
- **Important**: If you modify the dimension, ensure it is divisible by 8 (e.g., 128, 256, 512), as the AVX2 SIMD pre-filter processes floats in chunks of 8 (`__m256`).

### 2. Number of Clusters (centroids)
To adjust the clustering granularity:
- Open [build_index.py](file:///media/gultion/Material/Projects/Hybrid Vector Search Engine/scripts/build_index.py).
- Change `num_clusters` inside `build_index()`:
  ```python
  num_clusters = 1024  # Change to 2048 or 4096 for finer-grained clusters
  ```
- Note: Increasing the number of clusters decreases the average size of each cluster, which accelerates the GPU search but increases the CPU centroid pre-filtering cost.

### 3. Number of Searched Clusters (`top_n_clusters` / `nprobe`)
This is the primary knob for the **Latency vs. Recall (Accuracy)** trade-off:
- Open [benchmark.py](file:///media/gultion/Material/Projects/Hybrid Vector Search Engine/scripts/benchmark.py).
- Modify the `top_n_clusters` variable:
  ```python
  top_n_clusters = 8   # Low latency (19-60 ms), lower recall
  # top_n_clusters = 64  # Balanced (200 ms), high recall (40-60%)
  ```

### 4. Search Results Count ($K$)
To retrieve more or fewer nearest neighbors:
- Open [benchmark.py](file:///media/gultion/Material/Projects/Hybrid Vector Search Engine/scripts/benchmark.py).
- Modify the `k` variable:
  ```python
  k = 5                # Return top 5 nearest neighbors
  ```

### 5. Memory Allocation Caps (Host/GPU Pinned Buffers)
To avoid malloc overhead during search queries, the Hybrid Engine pre-allocates pinned memory during initialization. If you scale up search parameters, verify the bounds:
- Open [hybrid_engine.cpp](file:///media/gultion/Material/Projects/Hybrid Vector Search Engine/src/hybrid_engine.cpp).
- Modify these variables in the `HybridEngine` constructor:
  ```cpp
  max_candidates(500000) // Max candidate vectors allowed in the searched clusters (VRAM buffer size)
  max_k(256)             // Max search depth K supported by CUDA dynamic shared memory
  ```
- Memory requirement calculation for VRAM/Pinned RAM: 
  $$\text{Memory} = \text{max\_candidates} \times \text{dimension} \times 4 \text{ bytes}$$
  For `max_candidates = 500000` and `dimension = 512`, this allocates $\approx 1.024\text{ GB}$ of VRAM and $1.024\text{ GB}$ of pinned host RAM. If running on low system resources, you can decrease `max_candidates` to fit smaller budgets.
