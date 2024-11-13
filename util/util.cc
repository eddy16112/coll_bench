#include "util.h"
#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>

int allocate_memory_coll(void **buffer, size_t size, enum MemType type)
{
  size_t alignment = sysconf(_SC_PAGESIZE);
  int error = 0;

  switch (type) {
    case MemType::CPU:
      error = posix_memalign(buffer, alignment, size);
      //*buffer = malloc(sizeof(char) * size);
      memset(*buffer, 0, size);
      return error;
#ifdef _ENABLE_CUDA_
    case MemType::CUDA:
      CUDA_CHECK(cudaMalloc(buffer, size));
      return 0;
    case MANAGED:
      CUDA_CHECK(cudaMallocManaged(buffer, size, cudaMemAttachGlobal));
      return 0;
  #endif
    default:
      assert(0);
      return 1;
  }
}

void free_memory_coll(void *buffer, enum MemType type)
{
  switch (type) {
    case MemType::CPU:
      free(buffer);
      return;
#ifdef _ENABLE_CUDA_
    case MemType::CUDA:
      CUDA_CHECK(cudaFree(buffer));
      return;
    case MANAGED:
      CUDA_CHECK(cudaFree(buffer));
      return;
  #endif
    default:
      assert(0);
      return;
  }
}

void setup_buffer(enum DataType datatype, void *buf, int count, int rank)
{
  float *fbuf = nullptr;
  double *dbuf = nullptr;
  int *ibuf = nullptr;
  switch (datatype) {
    case DataType::FLOAT:
      fbuf = static_cast<float*>(buf);
      break;
    case DataType::DOUBLE:
      dbuf = static_cast<double*>(buf);
      break;
    case DataType::INT:
      ibuf = static_cast<int*>(buf);
      break;
  }
  for(int i = 0; i < count; i++) {
    if (fbuf) {
      fbuf[i] = rank;
    } else if (dbuf) {
      dbuf[i] = rank;
    } else if (ibuf) {
      ibuf[i] = rank;
    }
  }
}

bool validate_allreduce_result(enum DataType datatype, void *buf, int count, int rank, int comm_size)
{
  int expected = (0 + comm_size - 1) * comm_size / 2;
  float *fbuf = nullptr;
  double *dbuf = nullptr;
  int *ibuf = nullptr;
  switch (datatype) {
    case DataType::FLOAT:
      fbuf = static_cast<float*>(buf);
      break;
    case DataType::DOUBLE:
      dbuf = static_cast<double*>(buf);
      break;
    case DataType::INT:
      ibuf = static_cast<int*>(buf);
  }
  for(int i = 0; i < count; i++) {
    if (fbuf && fbuf[i] != static_cast<float>(expected)) {
      fprintf(stderr, "rank %d, expected %d, actual %f\n", rank, expected, fbuf[i]);
    } else if (dbuf && dbuf[i] != static_cast<double>(expected)) {
      fprintf(stderr, "rank %d, expected %d, actual %f\n", rank, expected, dbuf[i]);
    } else if (ibuf && ibuf[i] != static_cast<int>(expected)) {
      fprintf(stderr, "rank %d, expected %d, actual %d\n", rank, expected, ibuf[i]);
    }
  }
  return true;
}