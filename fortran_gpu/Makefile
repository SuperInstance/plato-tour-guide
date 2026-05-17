# Makefile for Fortran-style CUDA kernels
# Plato Tour Guide - Consensus System

# Fortran compiler
FC = gfortran
FFLAGS = -O3 -march=native -ffast-math -fcheck=no -Wall

# CUDA compiler
NVCC = nvcc
NVCCFLAGS = -O3 -arch=sm_80 -lineinfo

# Directories
SRC_DIR = .
BUILD_DIR = build
BIN_DIR = bin

# Targets
.PHONY: all clean fortran cuda benchmark test

all: fortran cuda benchmark

# Create build directories
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# Fortran targets
fortran: $(BIN_DIR)/cosine_kernel_test

$(BIN_DIR)/cosine_kernel_test: $(SRC_DIR)/cosine_kernel.f95 | $(BIN_DIR)
	$(FC) $(FFLAGS) $< -o $@
	@echo "Fortran kernel compiled successfully"
	@echo "Run with: $(BIN_DIR)/cosine_kernel_test"

# CUDA targets
cuda: $(BIN_DIR)/kernel_benchmark

$(BIN_DIR)/kernel_benchmark: $(SRC_DIR)/kernel_benchmark.cu | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $< -o $@
	@echo "CUDA benchmark compiled successfully"
	@echo "Run with: $(BIN_DIR)/kernel_benchmark [n_embeddings] [dim] [iters]"

# Test target
test: fortran
	@echo "Running Fortran kernel tests..."
	$(BIN_DIR)/cosine_kernel_test

# Benchmark target
benchmark: cuda
	@echo "Running CUDA kernel benchmarks..."
	$(BIN_DIR)/kernel_benchmark 1000 384 100

# Clean build artifacts
clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)
	rm -f *.mod *.o
	@echo "Cleaned build artifacts"

# Help target
help:
	@echo "Plato Tour Guide - Fortran-style CUDA Kernels"
	@echo ""
	@echo "Targets:"
	@echo "  all       - Build all targets (default)"
	@echo "  fortran   - Build Fortran kernel test"
	@echo "  cuda      - Build CUDA benchmark"
	@echo "  test      - Run Fortran kernel tests"
	@echo "  benchmark - Run CUDA benchmarks (1000 embeddings, 384 dim, 100 iters)"
	@echo "  clean     - Remove build artifacts"
	@echo "  help      - Show this help message"
	@echo ""
	@echo "Examples:"
	@echo "  make fortran           # Build Fortran kernel"
	@echo "  make test              # Run tests"
	@echo "  make benchmark         # Run benchmarks"
	@echo "  make cuda              # Build CUDA benchmark"
	@echo "  $(BIN_DIR)/kernel_benchmark 500 512 50  # Custom benchmark"
