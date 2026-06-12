import os

import numpy as np
from sklearn.cluster import MiniBatchKMeans


def build_index():
    base_file = "data/base_vectors.bin"
    centroids_file = "data/centroids.bin"
    vectors_file = "data/vectors.bin"

    if not os.path.exists(base_file):
        raise FileNotFoundError(
            f"Base vectors file {base_file} not found. Run generate_data.py first."
        )

    print(f"Reading base vectors from {base_file} via memmap...")
    # Read header first
    header = np.fromfile(base_file, dtype=np.uint32, count=2)
    num_vectors = int(header[0])
    dim = int(header[1])
    print(f"Dataset info: {num_vectors} vectors, dimension {dim}")

    # Memmap the base vectors to avoid loading all 2GB into RAM at once
    base_data = np.memmap(
        base_file, dtype=np.float32, mode="r", offset=8, shape=(num_vectors, dim)
    )

    num_clusters = 1024
    print(f"Clustering into {num_clusters} clusters using MiniBatchKMeans...")

    # We fit on a subset or the whole dataset in batches to be fast and memory-efficient
    kmeans = MiniBatchKMeans(
        n_clusters=num_clusters,
        batch_size=10000,
        max_no_improvement=20,
        random_state=42,
        n_init=3,
    )

    # Fit MiniBatchKMeans. To avoid high memory, we feed it in batches
    print("Fitting cluster centroids...")
    batch_size = 50000
    for i in range(0, num_vectors, batch_size):
        end_idx = min(i + batch_size, num_vectors)
        kmeans.partial_fit(base_data[i:end_idx])
        print(f"Clustered {end_idx}/{num_vectors} vectors...")

    centroids = kmeans.cluster_centers_.astype(np.float32)
    print(f"Centroids shape: {centroids.shape}")

    # Save centroids
    print(f"Saving centroids to {centroids_file}...")
    with open(centroids_file, "wb") as f:
        f.write(np.array([num_clusters, dim], dtype=np.uint32).tobytes())
        f.write(centroids.tobytes())

    print("Predicting cluster assignments...")
    cluster_ids = np.zeros(num_vectors, dtype=np.int32)
    for i in range(0, num_vectors, batch_size):
        end_idx = min(i + batch_size, num_vectors)
        cluster_ids[i:end_idx] = kmeans.predict(base_data[i:end_idx])

    print("Sorting indices by cluster assignment...")
    sorted_indices = np.argsort(cluster_ids).astype(np.uint32)

    # Calculate cluster sizes and offsets
    print("Calculating cluster sizes and offsets...")
    cluster_sizes = np.zeros(num_clusters, dtype=np.uint32)
    unique_ids, counts = np.unique(cluster_ids, return_counts=True)
    cluster_sizes[unique_ids] = counts.astype(np.uint32)

    cluster_offsets = np.zeros(num_clusters, dtype=np.uint32)
    current_offset = 0
    for c in range(num_clusters):
        cluster_offsets[c] = current_offset
        current_offset += cluster_sizes[c]

    print(f"Saving grouped vectors to {vectors_file} in chunked order...")
    # Open vectors.bin for writing
    with open(vectors_file, "wb") as f:
        # Write header: num_clusters (uint32), dimension (uint32), num_vectors (uint32)
        f.write(np.array([num_clusters, dim, num_vectors], dtype=np.uint32).tobytes())

        # Write cluster sizes
        f.write(cluster_sizes.tobytes())

        # Write cluster offsets
        f.write(cluster_offsets.tobytes())

        # Write sorted vector IDs
        f.write(sorted_indices.tobytes())

        # Write vector data sorted by cluster, writing in chunks to keep memory usage low
        write_chunk_size = 20000
        for i in range(0, num_vectors, write_chunk_size):
            end_idx = min(i + write_chunk_size, num_vectors)
            chunk_indices = sorted_indices[i:end_idx]
            # Gather vectors from memmap (numpy supports indexing of memmap, but doing it in chunks is efficient)
            chunk_data = base_data[chunk_indices]
            f.write(chunk_data.tobytes())

    print("Index build complete!")


if __name__ == "__main__":
    build_index()
