#include "util/util.h"
#include <stdio.h>
#include <assert.h>

MPI_Datatype get_mpi_datatype(enum DataType datatype);

MPI_Op get_mpi_op(enum Redop op);