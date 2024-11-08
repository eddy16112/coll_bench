#include "util/util.h"
#include <mpi.h>
#include <stdio.h>
#include <assert.h>

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

MPI_Datatype get_mpi_datatype(enum DataType datatype);

MPI_Op get_mpi_op(enum Redop op);