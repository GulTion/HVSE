import os
import numpy as np

def generate_and_save_data():
    os.makedirs('data', exist_ok=True)
    
    num_base = 1000000
    num_query = 100
    dim = 512
    
    print(f"Generating {num_base} base vectors of dimension {dim}...")
    # Generate vectors using normal distribution
    base_vectors = np.random.randn(num_base, dim).astype(np.float32)
    # L2 normalize them to make search meaningful, or just keep as-is (we'll keep as-is for Euclidean distance)
    
    base_file = 'data/base_vectors.bin'
    print(f"Saving base vectors to {base_file}...")
    with open(base_file, 'wb') as f:
        # Write header: num_vectors (uint32), dimension (uint32)
        f.write(np.array([num_base, dim], dtype=np.uint32).tobytes())
        # Write raw float32 data
        f.write(base_vectors.tobytes())
        
    print(f"Generating {num_query} query vectors of dimension {dim}...")
    query_vectors = np.random.randn(num_query, dim).astype(np.float32)
    
    query_file = 'data/query_vectors.bin'
    print(f"Saving query vectors to {query_file}...")
    with open(query_file, 'wb') as f:
        # Write header: num_vectors (uint32), dimension (uint32)
        f.write(np.array([num_query, dim], dtype=np.uint32).tobytes())
        # Write raw float32 data
        f.write(query_vectors.tobytes())
        
    print("Data generation complete!")

if __name__ == '__main__':
    generate_and_save_data()
