#include "mpi/util.h"

MPI_Datatype get_mpi_datatype(enum DataType datatype)
{
  switch (datatype) {
    case DataType::FLOAT:
      return MPI_FLOAT;
    case DataType::DOUBLE:
      return MPI_DOUBLE;
    case DataType::INT:
      return MPI_INT;
    default:
      return NULL;
  }
}

MPI_Op get_mpi_op(enum Redop op)
{
  switch (op) {
    case Redop::SUM:
      return MPI_SUM;
    default:
      return NULL;
  }
}