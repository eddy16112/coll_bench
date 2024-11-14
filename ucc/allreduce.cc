#include "ucc/util.h"
#include <stdlib.h>
#include <unistd.h>
#include <omp.h>
#include <numeric>

void run_allreduce(const AllreduceConfig &config, int num_threads, int tid)
{
  ucc_lib_h lib;
  create_ucc_lib(lib);
  UCCComm global_comm;

  create_ucc_comm(global_comm, tid, num_threads, lib);

  int rank = global_comm.global_rank;
  int comm_size = global_comm.global_comm_size;

  ucc_datatype_t datatype = get_ucc_datatype(config.datatype);
  ucc_reduction_op_t op = get_ucc_op(config.op);

  int datatype_size = get_ucc_datatype_size(config.datatype);

  void *sendbuf = nullptr;
  void *recvbuf = nullptr;
  size_t bufsize = config.count * datatype_size; 

  if (allocate_memory_coll((void **)&sendbuf, bufsize, MemType::CPU)) {
    fprintf(stderr, "Could Not Allocate sendbuf [rank %d]\n", rank);
    exit(1);
  }
  if (allocate_memory_coll((void **)&recvbuf, bufsize, MemType::CPU)) {
    fprintf(stderr, "Could Not Allocate recvbuf [rank %d]\n", rank);
    exit(1);
  }

  ucc_memory_type_t memtype = get_ucc_memtype(MemType::CPU);

  double t_start, t_stop, timer = 0;
  for (int i = 0; i < config.iteration + config.warmup; i++) {
    if (config.validate) {
      setup_buffer(config.datatype, sendbuf, config.count, rank);
    }
    t_start = MPI_Wtime();
    ucc_allreduce(sendbuf, memtype, recvbuf, memtype, config.count, datatype, op, global_comm);
    t_stop = MPI_Wtime();
   // MPI_CHECK(MPI_Barrier(comm));
    if (config.validate) {
      validate_allreduce_result(config.datatype, recvbuf, config.count, rank, comm_size);
    }
    if (i >= config.warmup) {
      timer += t_stop - t_start;
    }
    printf("done with iter %d\n", i);
  }

  double latency = static_cast<double>(timer * 1e6) / config.iteration;
  double avg_time = 0.0, max_time = 0.0, min_time = 0.0;

  // MPI_CHECK(MPI_Reduce(&latency, &min_time, 1, MPI_DOUBLE, MPI_MIN, 0, comm));
  // MPI_CHECK(MPI_Reduce(&latency, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, comm));
  // MPI_CHECK(MPI_Reduce(&latency, &avg_time, 1, MPI_DOUBLE, MPI_SUM, 0, comm));
  // if (rank == 0) {
  //   avg_time = avg_time / comm_size;
  //   printf("allreduce size %lu, avg %f, min %f, max %f\n", bufsize, avg_time, min_time, max_time);
  // }

  free_memory_coll(sendbuf, MemType::CPU);
  free_memory_coll(recvbuf, MemType::CPU);
  destroy_ucc_comm(global_comm);
  ucc_finalize(lib);
}

int main(int argc, char **argv)
{
  int provided = 0;
  MPI_CHECK(MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided));

  int num_threads = 1;
  printf("pid %d\n", getpid());
    // sleep(10);
  
  AllreduceConfig config{10, 10, 262144, true, DataType::INT, MemType::CPU, Redop::SUM};
  if (num_threads > 1) {
#pragma omp parallel num_threads(num_threads)
{
    int tid = omp_get_thread_num();
    run_allreduce(config, num_threads, tid);
}
  } else {
    run_allreduce(config, 1, 0);
  }

  MPI_CHECK(MPI_Finalize());
  return 0;
}