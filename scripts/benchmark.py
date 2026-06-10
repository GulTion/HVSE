import os
import subprocess
import time
import numpy as np

def run_cpp_binary(cmd):
    print(f"Running: {' '.join(cmd)}")
    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    end = time.time()
    
    if result.returncode != 0:
        print(f"Error running command: {result.stderr}")
        return None, None
        
    # Parse output to extract results and mean search latency
    stdout = result.stdout
    lines = stdout.split('\n')
    
    latency = None
    query_results = {}
    current_query = None
    
    for line in lines:
        if "Mean search latency:" in line:
            # Extract number
            parts = line.split("Mean search latency:")
            latency = float(parts[1].strip().split()[0])
        elif "Query" in line and "Results" in line:
            current_query = int(line.split()[1])
            query_results[current_query] = []
        elif "Rank" in line and current_query is not None:
            # Rank X: ID=Y, Distance=Z
            parts = line.split("ID=")
            id_part = parts[1].split(",")[0]
            dist_part = parts[1].split("Distance=")[1]
            query_results[current_query].append((int(id_part), float(dist_part)))
            
    return latency, query_results

def run_benchmarks():
    base_file = 'data/base_vectors.bin'
    query_file = 'data/query_vectors.bin'
    centroids_file = 'data/centroids.bin'
    vectors_file = 'data/vectors.bin'
    
    k = 5
    num_queries = 10
    top_n_clusters = 8
    
    # 1. Run C++ Naive CPU search
    naive_cmd = ['./cpu_naive', base_file, query_file, str(k), str(num_queries)]
    naive_latency, naive_results = run_cpp_binary(naive_cmd)
    
    # 2. Run C++ Hybrid Search
    hybrid_cmd = ['./hybrid_search', centroids_file, vectors_file, query_file, str(k), str(num_queries), str(top_n_clusters)]
    hybrid_latency, hybrid_results = run_cpp_binary(hybrid_cmd)
    
    # Check if PyTorch / Faiss can be run (i.e. on Colab)
    pytorch_latency = None
    faiss_latency = None
    
    # Load data for Python benchmarks if libraries are present
    has_torch = False
    has_faiss = False
    
    try:
        import torch
        has_torch = True
    except ImportError:
        pass
        
    try:
        import faiss
        has_faiss = True
    except ImportError:
        pass
        
    if has_torch or has_faiss:
        print("\nLoading vectors into Python memory for PyTorch/Faiss benchmarks...")
        # Read headers
        base_hdr = np.fromfile(base_file, dtype=np.uint32, count=2)
        n_base = int(base_hdr[0])
        dim = int(base_hdr[1])
        base_vectors = np.memmap(base_file, dtype=np.float32, mode='r', offset=8, shape=(n_base, dim))
        
        query_hdr = np.fromfile(query_file, dtype=np.uint32, count=2)
        n_query = int(query_hdr[0])
        query_vectors = np.memmap(query_file, dtype=np.float32, mode='r', offset=8, shape=(n_query, dim))
        
        # Limit to num_queries
        queries_subset = query_vectors[:num_queries]
        
        if has_torch:
            print("Running PyTorch GPU cdist benchmark...")
            if torch.cuda.is_available():
                try:
                    # Warm-up
                    q_dev = torch.from_numpy(queries_subset).cuda()
                    # To prevent OOM on low VRAM GPUs, process base_vectors in chunks
                    # On Colab (16GB VRAM), we can do it directly.
                    # We will implement a chunked cdist to run safely on any GPU
                    chunk_size = 100000
                    
                    # Warm-up
                    temp_base = torch.from_numpy(base_vectors[:10000]).cuda()
                    torch.cdist(q_dev, temp_base)
                    
                    torch.cuda.synchronize()
                    start_time = time.time()
                    
                    for q_idx in range(num_queries):
                        q_single = q_dev[q_idx : q_idx + 1]
                        min_dists = []
                        min_indices = []
                        for i in range(0, n_base, chunk_size):
                            end_i = min(i + chunk_size, n_base)
                            b_chunk = torch.from_numpy(base_vectors[i:end_i]).cuda()
                            dists = torch.cdist(q_single, b_chunk) # shape (1, chunk_size)
                            val, idx = torch.topk(dists, k, largest=False)
                            min_dists.append(val.cpu())
                            min_indices.append(idx.cpu() + i)
                            
                        # Merge chunk results
                        all_dists = torch.cat(min_dists, dim=1) # (1, num_chunks * k)
                        all_indices = torch.cat(min_indices, dim=1)
                        final_val, final_idx = torch.topk(all_dists, k, largest=False)
                        
                    torch.cuda.synchronize()
                    end_time = time.time()
                    pytorch_latency = ((end_time - start_time) / num_queries) * 1000.0 # ms
                    print(f"PyTorch GPU Mean search latency: {pytorch_latency:.3f} ms")
                except Exception as e:
                    print(f"PyTorch GPU run failed: {e}")
            else:
                print("PyTorch CUDA not available. Skipping PyTorch GPU benchmark.")
                
        if has_faiss:
            print("Running Faiss IVF-Flat CPU benchmark...")
            try:
                # Build a simple IVF-Flat index
                quantizer = faiss.IndexFlatL2(dim)
                index = faiss.IndexIVFFlat(quantizer, dim, top_n_clusters)
                # Read centroids and train
                centroids_hdr = np.fromfile(centroids_file, dtype=np.uint32, count=2)
                n_centroids = int(centroids_hdr[0])
                centroids = np.fromfile(centroids_file, dtype=np.float32, offset=8).reshape(n_centroids, dim)
                
                # Train the index with centroids
                index.train(centroids)
                # Add base vectors in chunks to save memory
                add_chunk = 100000
                for i in range(0, n_base, add_chunk):
                    end_i = min(i + add_chunk, n_base)
                    index.add(base_vectors[i:end_i])
                    
                # Search
                index.nprobe = top_n_clusters
                # Warm-up
                index.search(queries_subset[:1], k)
                
                start_time = time.time()
                D, I = index.search(queries_subset, k)
                end_time = time.time()
                
                faiss_latency = ((end_time - start_time) / num_queries) * 1000.0 # ms
                print(f"Faiss IVF-Flat Mean search latency: {faiss_latency:.3f} ms")
            except Exception as e:
                print(f"Faiss run failed: {e}")

    # 3. Print verification table
    print("\n" + "="*50)
    print("CORRECTNESS AND RECALL VERIFICATION")
    print("="*50)
    if naive_results and hybrid_results:
        recalls = []
        for q in range(min(num_queries, 5)):
            if q in naive_results and q in hybrid_results:
                naive_ids = set([pair[0] for pair in naive_results[q]])
                hybrid_ids = set([pair[0] for pair in hybrid_results[q]])
                intersection = naive_ids.intersection(hybrid_ids)
                recall = len(intersection) / k
                recalls.append(recall)
                print(f"Query {q}: Naive IDs={sorted(list(naive_ids))}, Hybrid IDs={sorted(list(hybrid_ids))}, Recall={recall*100:.1f}%")
        if recalls:
            print(f"Average top-{k} Recall (IVF vs Exact CPU): {np.mean(recalls)*100:.1f}%")

    print("\n" + "="*50)
    print("PERFORMANCE SUMMARY")
    print("="*50)
    print(f"{'Engine/Baseline':<25} | {'Mean Latency':<15}")
    print("-"*45)
    if naive_latency is not None:
        print(f"{'Baseline 1: Naive CPU':<25} | {naive_latency:.3f} ms")
    if pytorch_latency is not None:
        print(f"{'Baseline 2: PyTorch GPU':<25} | {pytorch_latency:.3f} ms")
    if faiss_latency is not None:
        print(f"{'Faiss IVF-Flat':<25} | {faiss_latency:.3f} ms")
    if hybrid_latency is not None:
        print(f"{'Our Hybrid Engine':<25} | {hybrid_latency:.3f} ms")
        
    if naive_latency and hybrid_latency:
        speedup = naive_latency / hybrid_latency
        print(f"\nSpeedup of Hybrid Engine vs. Naive CPU: {speedup:.1f}x")

if __name__ == '__main__':
    run_benchmarks()
