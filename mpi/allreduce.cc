#include "mpi/util.h"
#include <stdlib.h>
#include <unistd.h>

void run_allreduce(const Config &config)
{
  MPI_Comm comm = *((MPI_Comm*)(config.comm));
  int rank, comm_size;
  MPI_CHECK(MPI_Comm_rank(comm, &rank));
  MPI_CHECK(MPI_Comm_size(comm, &comm_size));

  MPI_Datatype datatype = get_mpi_datatype(config.datatype);
  MPI_Op op = get_mpi_op(config.op);

  int mpi_type_size;
  MPI_CHECK(MPI_Type_size(datatype, &mpi_type_size));

  void *sendbuf = nullptr;
  void *recvbuf = nullptr;
  size_t bufsize = config.count * mpi_type_size; 

  if (allocate_memory_coll((void **)&sendbuf, bufsize, MemType::CPU)) {
    fprintf(stderr, "Could Not Allocate sendbuf [rank %d]\n", rank);
    MPI_CHECK(MPI_Abort(comm, EXIT_FAILURE));
  }
  if (allocate_memory_coll((void **)&recvbuf, bufsize, MemType::CPU)) {
    fprintf(stderr, "Could Not Allocate recvbuf [rank %d]\n", rank);
    MPI_CHECK(MPI_Abort(comm, EXIT_FAILURE));
  }
  
  MPI_Barrier(comm);

  double t_start, t_stop, timer = 0;
  for (int i = 0; i < config.iteration + config.warmup; i++) {
    if (config.validate) {
      setup_buffer(config.datatype, sendbuf, config.count, rank);
    }
    t_start = MPI_Wtime();
    MPI_CHECK(MPI_Allreduce(sendbuf, recvbuf, config.count,
                            datatype, op, comm));
    t_stop = MPI_Wtime();
    MPI_CHECK(MPI_Barrier(comm));
    if (config.validate) {
      validate_allreduce_result(config.datatype, recvbuf, config.count, rank, comm_size);
    }
    if (i >= config.warmup) {
      timer += t_stop - t_start;
    }
  }

  double latency = static_cast<double>(timer * 1e6) / config.iteration;
  double avg_time = 0.0, max_time = 0.0, min_time = 0.0;

  MPI_CHECK(MPI_Reduce(&latency, &min_time, 1, MPI_DOUBLE, MPI_MIN, 0, comm));
  MPI_CHECK(MPI_Reduce(&latency, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, comm));
  MPI_CHECK(MPI_Reduce(&latency, &avg_time, 1, MPI_DOUBLE, MPI_SUM, 0, comm));
  if (rank == 0) {
    avg_time = avg_time / comm_size;
    printf("allreduce size %lu, avg %f, min %f, max %f\n", bufsize, avg_time, min_time, max_time);
  }


}

int main(int argc, char **argv)
{
  int provided = 0;
  MPI_CHECK(MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided));

  MPI_Comm comm = MPI_COMM_WORLD;
  printf("pid %d\n", getpid());
  //sleep(10);

  Config config{1000, 10, 262144, false, DataType::INT, MemType::CPU, Redop::SUM, &comm};
  run_allreduce(config);

  MPI_CHECK(MPI_Finalize());

}
