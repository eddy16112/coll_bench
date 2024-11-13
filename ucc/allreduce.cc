#include "ucc/util.h"
#include <stdlib.h>
#include <unistd.h>
#include <omp.h>
#include <numeric>

static void bcast(void *sendbuf, void *recvbuf, size_t count, ucc_datatype_t type, ucc_reduction_op_t op, UCCComm &comm)
{
    ucc_coll_req_h req;

    ucc_coll_args_t coll = { 0 };
    coll.mask = UCC_COLL_ARGS_FIELD_FLAGS;
    coll.flags = UCC_COLL_ARGS_FLAG_COUNT_64BIT;
    coll.coll_type = UCC_COLL_TYPE_ALLREDUCE;
    coll.op = op;

    coll.src.info.buffer = sendbuf;
    coll.src.info.count = count;
    coll.src.info.datatype = type;
    coll.src.info.mem_type = UCC_MEMORY_TYPE_HOST;

    coll.dst.info.buffer = recvbuf;
    coll.dst.info.count = count;
    coll.dst.info.datatype = type;
    coll.dst.info.mem_type = UCC_MEMORY_TYPE_HOST;

    UCC_CHECK(ucc_collective_init(&coll, &req, comm.team));
    UCC_CHECK(ucc_collective_post(req));

    while (ucc_collective_test(req) > UCC_OK) {
        UCC_CHECK(ucc_context_progress(comm.ctx));
    }

    UCC_CHECK(ucc_collective_finalize(req));
}

int main(int argc, char **argv)
{
  int provided = 0;
  MPI_CHECK(MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided));

      int mpi_rank, mpi_size;
    MPI_Comm mpi_comm = MPI_COMM_WORLD;
    MPI_Comm_rank(mpi_comm, &mpi_rank);
    MPI_Comm_size(mpi_comm, &mpi_size);

    int num_threads = 10;
    printf("pid %d\n", getpid());
    // sleep(10);

#pragma omp parallel num_threads(num_threads)
{
    int tid = omp_get_thread_num();

    // creating communicator handle with MPI communicator
    int size = mpi_size * num_threads;
    int rank = num_threads * mpi_rank + tid;

    UCCComm global_comm;
    global_comm.mpi_comm_size = mpi_size;
    global_comm.mpi_rank = mpi_rank;
    global_comm.nb_threads = num_threads;
    global_comm.tid = tid;
    global_comm.global_rank = rank;
    global_comm.comm = MPI_COMM_WORLD;
    global_comm.global_comm_size = size;

    ucc_context_h ctx;
    ucc_team_h team;
    ucc_lib_h lib;

    create_ucc_lib(lib);
    create_ucc_cxt(global_comm, lib, ctx);
    std::vector<int> top_rank_mapping;
    top_rank_mapping.resize(size);
    std::iota(top_rank_mapping.begin(), top_rank_mapping.end(), 0);
    create_ucc_team(rank, size, top_rank_mapping, ctx, team);

    global_comm.ctx = ctx;
    global_comm.team = team;

    // create_ucc_comm(global_comm, tid, num_threads,)

    int32_t *sendbuf = nullptr;
    int32_t *recvbuf = nullptr;
    size_t send_size = 128;
    sendbuf = (int32_t *)malloc(sizeof(int32_t) * send_size);
    recvbuf = (int32_t *)malloc(sizeof(int32_t) * send_size);
    for (size_t i = 0; i < send_size; i++) {
        sendbuf[i] = rank;
        recvbuf[i] = 0;
    }

    bcast(sendbuf, recvbuf, send_size, UCC_DT_INT32, UCC_OP_SUM, global_comm);

    int expected = (0 + size - 1) * size / 2;
    for (size_t i = 0; i < send_size; i++) {
        if (recvbuf[i] != expected) {
            printf("error rank %d, tid %d, val %d\n", rank, tid, recvbuf[i]);
            assert(0);
        }
    } 

    free(sendbuf);
    free(recvbuf);
    printf("done with bcast rank %d\n", rank);
    destroy_ucc_comm(global_comm);
    ucc_finalize(lib);
}

  MPI_CHECK(MPI_Finalize());
  return 0;
}