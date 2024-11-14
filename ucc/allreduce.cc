#include "ucc/util.h"
#include <stdlib.h>
#include <unistd.h>
#include <omp.h>
#include <numeric>

void run_allreduce(const AllreduceConfig &config, int num_threads, int tid, ucc_lib_h lib)
{
  UCCComm comm;

  create_ucc_comm(comm, tid, num_threads, lib);

  int rank = comm.rank;
  int comm_size = comm.comm_size;

  ucc_datatype_t datatype = get_ucc_datatype(config.datatype);
  ucc_reduction_op_t op = get_ucc_op(config.op);

  int datatype_size = get_ucc_datatype_size(config.datatype);

  void *sendbuf = nullptr;
  void *recvbuf = nullptr;
  size_t bufsize = config.count * datatype_size; 

  if (allocate_memory_coll((void **)&sendbuf, bufsize, config.memtype)) {
    fprintf(stderr, "Could Not Allocate sendbuf [rank %d]\n", rank);
    exit(1);
  }
  if (allocate_memory_coll((void **)&recvbuf, bufsize, config.memtype)) {
    fprintf(stderr, "Could Not Allocate recvbuf [rank %d]\n", rank);
    exit(1);
  }

  ucc_memory_type_t memtype = get_ucc_memtype(config.memtype);

  ucc_barrier(comm);

  double t_start, t_stop, timer = 0;
  for (int i = 0; i < config.iteration + config.warmup; i++) {
    if (config.validate) {
      setup_buffer(config.datatype, sendbuf, config.count, rank);
    }
    t_start = MPI_Wtime();
    ucc_allreduce(sendbuf, memtype, recvbuf, memtype, config.count, datatype, op, comm);
    t_stop = MPI_Wtime();
    ucc_barrier(comm);
    if (config.validate) {
      validate_allreduce_result(config.datatype, recvbuf, config.count, rank, comm_size);
    }
    if (i >= config.warmup) {
      timer += t_stop - t_start;
    }
  }

  double latency = static_cast<double>(timer * 1e6) / config.iteration;
  double avg_time = 0.0, max_time = 0.0, min_time = 0.0;
  ucc_reduce(&latency, UCC_MEMORY_TYPE_HOST, &min_time, UCC_MEMORY_TYPE_HOST, 1, UCC_DT_FLOAT64, UCC_OP_MIN, 0, comm);
  ucc_reduce(&latency, UCC_MEMORY_TYPE_HOST, &max_time, UCC_MEMORY_TYPE_HOST, 1, UCC_DT_FLOAT64, UCC_OP_MAX, 0, comm);
  ucc_reduce(&latency, UCC_MEMORY_TYPE_HOST, &avg_time, UCC_MEMORY_TYPE_HOST, 1, UCC_DT_FLOAT64, UCC_OP_SUM, 0, comm);
  if (rank == 0) {
    avg_time = avg_time / comm_size;
    printf("allreduce size %lu, avg %f, min %f, max %f\n", bufsize, avg_time, min_time, max_time);
  }

  free_memory_coll(sendbuf, config.memtype);
  free_memory_coll(recvbuf, config.memtype);
  destroy_ucc_comm(comm);
}

int main(int argc, char **argv)
{
  int provided = 0;
  MPI_CHECK(MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided));

  int num_threads = 10;
  printf("pid %d\n", getpid());
    // sleep(10);
  
  AllreduceConfig config{10, 10, 262144, true, DataType::INT, MemType::CPU, Redop::SUM};
  if (num_threads > 1) {
#pragma omp parallel num_threads(num_threads)
{
    ucc_lib_h lib;
    create_ucc_lib(lib);
    int tid = omp_get_thread_num();
    run_allreduce(config, num_threads, tid, lib);
    ucc_finalize(lib);
}
  } else {
    ucc_lib_h lib;
    create_ucc_lib(lib);
    run_allreduce(config, 1, 0, lib);
    ucc_finalize(lib);
  }

  MPI_CHECK(MPI_Finalize());
  return 0;
}