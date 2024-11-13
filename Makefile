# Compiler and flags
CXX := mpicxx
CXXFLAGS := -std=c++11 -O2 -fopenmp
LDFLAGS :=

ROOT_DIR := /scratch2/wwu/coll_bench
UCC_LIB_DIR := /scratch2/wwu/ucc-mtcuda/install
UCX_LIB_DIR := /scratch2/wwu/ucx-1.15.0

# Include directories
INC_FLAGS := -I$(ROOT_DIR) -I$(UCC_LIB_DIR)/include -I$(UCX_LIB_DIR)/include
LDFLAGS += -L$(UCC_LIB_DIR)/lib -lucc -L$(UCX_LIB_DIR)/lib -lucp

# Directories
MPI_DIR := mpi
UCC_DIR := ucc
UTIL_DIR := util
BIN_DIR := bin

# Source files
MPI_SRCS := $(MPI_DIR)/allreduce.cc
UCC_SRCS := $(UCC_DIR)/allreduce.cc
UTIL_SRCS := $(UTIL_DIR)/util.cc

# Object files
MPI_UTIL_OBJ := $(MPI_DIR)/util.o
UCC_UTIL_OBJ := $(UCC_DIR)/util.o
UTIL_OBJ := $(UTIL_DIR)/util.o

# Binaries
BINARIES := $(BIN_DIR)/mpi_allreduce $(BIN_DIR)/ucc_allreduce

# Target to build all binaries
all: $(BIN_DIR) $(BINARIES)

# Create bin directory if it doesn't exist
$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# Compile allreduce binary
$(BIN_DIR)/mpi_allreduce: $(MPI_DIR)/allreduce.o $(UTIL_OBJ) $(MPI_UTIL_OBJ)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

$(BIN_DIR)/ucc_allreduce: $(UCC_DIR)/allreduce.o $(UTIL_OBJ) $(UCC_UTIL_OBJ)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

# Compile object files
$(MPI_DIR)/%.o: $(MPI_DIR)/%.cc $(UTIL_DIR)/util.h
	$(CXX) $(CXXFLAGS) $(INC_FLAGS) -c $< -o $@

$(UCC_DIR)/%.o: $(UCC_DIR)/%.cc $(UTIL_DIR)/util.h
	$(CXX) $(CXXFLAGS) $(INC_FLAGS) -c $< -o $@

$(UTIL_OBJ): $(UTIL_SRCS) $(UTIL_DIR)/util.h
	$(CXX) $(CXXFLAGS) $(INC_FLAGS) -c $< -o $@

# Clean up
clean:
	rm -rf $(MPI_DIR)/*.o $(UCC_DIR)/*.o $(UTIL_OBJ) $(BIN_DIR)

.PHONY: all clean
