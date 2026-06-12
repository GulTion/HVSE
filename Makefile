# Compiler configurations
CXX = g++
NVCC = nvcc

# CUDA Installation path (default is /usr/local/cuda)
CUDA_PATH ?= /usr/local/cuda
CUDA_LIB = -L$(CUDA_PATH)/lib64 -lcudart

# Compilation flags
CXXFLAGS = -O3 -std=c++17 -mavx2 -mfma -Wall -Wextra -Isrc -I$(CUDA_PATH)/include
NVCCFLAGS = -O3 -std=c++17 -Isrc

# GPU architecture flags
# Default to local MX350 (Pascal CC 6.1). Switch to Turing CC 7.5 for Colab T4.
ifeq ($(TARGET), colab)
    GPU_ARCH = -gencode arch=compute_75,code=sm_75
    $(info Compiling for Colab T4 (Compute 7.5))
else
    GPU_ARCH = -gencode arch=compute_61,code=sm_61
    $(info Compiling for local MX350 (Compute 6.1))
endif

NVCCFLAGS += $(GPU_ARCH)

# Output binaries
NAIVE_BIN = cpu_naive
HYBRID_BIN = hybrid_search
INDEX_BIN = build_index_cuda

# Source and Object files
SRC_DIR = src
OBJ_DIR = obj

# Create directories if they do not exist
$(shell mkdir -p $(OBJ_DIR))

# Default target
all: $(NAIVE_BIN) $(HYBRID_BIN) $(INDEX_BIN)

# Link Naive CPU baseline
$(NAIVE_BIN): $(OBJ_DIR)/cpu_naive.o
	$(CXX) $(CXXFLAGS) -o $@ $^

# Link Hybrid Search Engine (Host code + CUDA device code)
$(HYBRID_BIN): $(OBJ_DIR)/main.o $(OBJ_DIR)/hybrid_engine.o $(OBJ_DIR)/kernel.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(CUDA_LIB)

# Link CUDA Index Builder
$(INDEX_BIN): $(OBJ_DIR)/build_index_cuda.o
	$(NVCC) $(NVCCFLAGS) -o $@ $^

# Compile C++ source files
$(OBJ_DIR)/cpu_naive.o: $(SRC_DIR)/cpu_naive.cpp $(SRC_DIR)/common.h
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(OBJ_DIR)/hybrid_engine.o: $(SRC_DIR)/hybrid_engine.cpp $(SRC_DIR)/hybrid_engine.h $(SRC_DIR)/common.h
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(OBJ_DIR)/main.o: $(SRC_DIR)/main.cpp $(SRC_DIR)/hybrid_engine.h $(SRC_DIR)/common.h
	$(CXX) $(CXXFLAGS) -c -o $@ $<

# Compile CUDA source files
$(OBJ_DIR)/kernel.o: $(SRC_DIR)/kernel.cu $(SRC_DIR)/common.h
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

$(OBJ_DIR)/build_index_cuda.o: $(SRC_DIR)/build_index_cuda.cu $(SRC_DIR)/common.h
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

# Clean build artifacts
clean:
	rm -rf $(OBJ_DIR) $(NAIVE_BIN) $(HYBRID_BIN) $(INDEX_BIN)

.PHONY: all clean
