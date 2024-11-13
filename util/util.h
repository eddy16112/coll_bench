
#include <stddef.h>
#include <mpi.h>

#define EXIT_FAILURE

enum MemType {CPU, CUDA, MANAGED};

enum DataType {FLOAT, DOUBLE, INT};

enum Redop {SUM};

struct AllreduceConfig
{
  int iteration;
  int warmup;
  int count;
  bool validate;
  enum DataType datatype;
  enum MemType memtype;
  enum Redop op;
};

#define MPI_CHECK(stmt)                                                  \
  do {                                                                   \
    int mpi_errno = (stmt);                                              \
    if (MPI_SUCCESS != mpi_errno) {                                      \
      fprintf(stderr, "[%s:%d] MPI call failed with %d \n", __FILE__,    \
              __LINE__, mpi_errno);                                      \
      exit(EXIT_FAILURE);                                                \
    }                                                                    \
    assert(MPI_SUCCESS == mpi_errno);                                    \
  } while (0)

int allocate_memory_coll(void **buffer, size_t size, enum MemType type);

void free_memory_coll(void *buffer, enum MemType type);

void setup_buffer(enum DataType datatype, void *buf, int count, int rank);

bool validate_allreduce_result(enum DataType datatype, void *buf, int count, int rank, int comm_size);