# Compiler and flags
CXX := mpicxx
CXXFLAGS := -std=c++11 -O2
LDFLAGS :=

ROOT_DIR := /scratch2/wwu/coll_bench

# Include directories
INCLUDES := -I$(ROOT_DIR)

# Directories
MPI_DIR := mpi
UTIL_DIR := util
BIN_DIR := bin

# Source files
MPI_SRCS := $(MPI_DIR)/allreduce.cc
UTIL_SRCS := $(UTIL_DIR)/util.cc

# Object files
MPI_UTIL_OBJ := $(MPI_DIR)/util.o
UTIL_OBJ := $(UTIL_DIR)/util.o

# Binaries
BINARIES := $(BIN_DIR)/mpi_allreduce

# Target to build all binaries
all: $(BIN_DIR) $(BINARIES)

# Create bin directory if it doesn't exist
$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# Compile allreduce binary
$(BIN_DIR)/mpi_allreduce: $(MPI_DIR)/allreduce.o $(UTIL_OBJ) $(MPI_UTIL_OBJ)
	$(CXX) $^ -o $@ $(LDFLAGS)

# Compile object files
$(MPI_DIR)/%.o: $(MPI_DIR)/%.cc $(UTIL_DIR)/util.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(UTIL_OBJ): $(UTIL_SRCS) $(UTIL_DIR)/util.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Clean up
clean:
	rm -rf $(MPI_DIR)/*.o $(UTIL_OBJ) $(BIN_DIR)

.PHONY: all clean
