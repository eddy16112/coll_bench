
#include <stddef.h>

#define EXIT_FAILURE

enum MemType {CPU, CUDA, MANAGED};

enum DataType {FLOAT, DOUBLE, INT};

enum Redop {SUM};

struct Config
{
  int iteration;
  int warmup;
  int count;
  bool validate;
  enum DataType datatype;
  enum MemType memtype;
  enum Redop op;
  void *comm;
};

int allocate_memory_coll(void **buffer, size_t size, enum MemType type);

void setup_buffer(enum DataType datatype, void *buf, int count, int rank);

bool validate_allreduce_result(enum DataType datatype, void *buf, int count, int rank, int comm_size);