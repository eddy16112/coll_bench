#include "ucc/util.h"
#include <cstring>
#include <numeric>

static int Coll_Gather_thread(void *sendbuf, int sendcount, MPI_Datatype sendtype, 
                              void *recvbuf, int recvcount, MPI_Datatype recvtype, 
                              int root,
                              UCCComm global_comm)
{	
  int res;

  int total_size = global_comm.mpi_comm_size * global_comm.nb_threads;
  MPI_Status status;
 
  int global_rank = global_comm.global_rank;
  assert(global_rank / global_comm.nb_threads == global_comm.mpi_rank);
  assert(global_rank == global_comm.mpi_rank * global_comm.nb_threads + global_comm.tid);

  // MPI_IN_PLACE
  if (sendbuf == recvbuf) {
    assert(0);
  }

  int root_mpi_rank = root / global_comm.nb_threads;

  int tag;

  assert(global_comm.starting_tag >= 0);
  
  // non-root
  if (global_rank != root) {
    tag = global_comm.starting_tag * 10000 + global_rank;
    return MPI_Send(sendbuf, sendcount, sendtype, root_mpi_rank, tag, global_comm.comm);
  } 

  // root
  MPI_Aint incr, lb, recvtype_extent;
  MPI_Type_get_extent(recvtype, &lb, &recvtype_extent);
  incr = recvtype_extent * (ptrdiff_t)recvcount;
  char *dst = (char*)recvbuf;
  int recvfrom_mpi_rank;
  for(int i = 0 ; i < total_size; i++) {
    recvfrom_mpi_rank = i / global_comm.nb_threads;
    tag = global_comm.starting_tag * 10000 + i;
    assert(dst != NULL);
    if (global_rank == i) {
      memcpy(dst, sendbuf, incr);
    } else {
      res = MPI_Recv(dst, recvcount, recvtype, recvfrom_mpi_rank, tag, global_comm.comm, &status);
      assert(res == MPI_SUCCESS);
    }
    dst += incr;
	}

  return 0;
}

static int Coll_Allgather_thread(void *sendbuf, int sendcount, MPI_Datatype sendtype, 
                                 void *recvbuf, int recvcount, MPI_Datatype recvtype, 
                                 UCCComm global_comm)
{	
  int total_size = global_comm.mpi_comm_size * global_comm.nb_threads;

  printf("allgather thread\n");

  // MPI_IN_PLACE
  if (sendbuf == recvbuf) {
    assert(0);
  }
  
  int global_rank = global_comm.mpi_rank * global_comm.nb_threads + global_comm.tid;
  for(int i = 0 ; i < total_size; i++) {
    // printf("global_rank %d, i %d\n", global_rank, i);
    global_comm.starting_tag = i;
    Coll_Gather_thread(sendbuf, sendcount, sendtype, 
                       recvbuf, recvcount, recvtype, 
                       i, global_comm);
	}
  return 0;
}

static ucc_status_t allgather(void *sbuf, void *rbuf, size_t msglen,
                              void *coll_info, void **req)
{
  MPI_Comm    comm = ((UCCComm *)coll_info)->comm;
  MPI_Request request;
  int error = MPI_Allgather(sbuf, msglen, MPI_BYTE, rbuf, msglen, MPI_BYTE, comm);
  assert(error == MPI_SUCCESS);
  *req = (void *)(uintptr_t)request;
  return UCC_OK;
}

static ucc_status_t allgather_thread(void *sbuf, void *rbuf, size_t msglen,
                                     void *coll_info, void **req)
{
  UCCComm    *comm = (UCCComm *)coll_info;
  MPI_Request request;
  int error = Coll_Allgather_thread(sbuf, msglen, MPI_BYTE, rbuf, msglen, MPI_BYTE, *comm);
  assert(error == 0);
  *req = (void *)(uintptr_t)request;
  return UCC_OK;
}

static ucc_status_t request_test(void *req)
{
  return UCC_OK;
}

static ucc_status_t request_free(void *req)
{
  return UCC_OK;
}

void create_ucc_lib(ucc_lib_h &lib)
{   
  ucc_lib_config_h lib_config;
  ucc_lib_params_t lib_params = { 0 };
  lib_params.mask = UCC_LIB_PARAM_FIELD_THREAD_MODE;
  lib_params.thread_mode = UCC_THREAD_MULTIPLE;
  UCC_CHECK(ucc_lib_config_read(NULL, NULL, &lib_config));
  UCC_CHECK(ucc_init(&lib_params, lib_config, &lib));
  ucc_lib_config_release(lib_config);
}

void create_ucc_cxt(UCCComm &comm, ucc_lib_h lib, ucc_context_h &ctx)
{
  ucc_context_config_h ctx_config;
  ucc_context_params_t ctx_params = { 0 };
  ctx_params.mask = UCC_CONTEXT_PARAM_FIELD_TYPE | UCC_CONTEXT_PARAM_FIELD_OOB;
  ctx_params.type = UCC_CONTEXT_SHARED;
  ctx_params.oob.allgather = allgather_thread;
  ctx_params.oob.req_test = request_test;
  ctx_params.oob.req_free = request_free;
  ctx_params.oob.coll_info = (void*)(&comm);
  ctx_params.oob.n_oob_eps = comm.global_comm_size;
  ctx_params.oob.oob_ep = comm.global_rank;
  UCC_CHECK(ucc_context_config_read(lib, NULL, &ctx_config));
  UCC_CHECK(ucc_context_create(lib, &ctx_params, ctx_config, &ctx));
  ucc_context_config_release(ctx_config);
}

void create_ucc_team(int rank, int nranks, std::vector<int> &rank_mapping, ucc_context_h ctx, ucc_team_h &team)
{
    ucc_team_params_t team_params = { 0 };
    team_params.mask =
        UCC_TEAM_PARAM_FIELD_EP_MAP | UCC_TEAM_PARAM_FIELD_EP | UCC_TEAM_PARAM_FIELD_EP_RANGE | UCC_TEAM_PARAM_FIELD_ID;
    team_params.ep_map.type = UCC_EP_MAP_ARRAY;
    team_params.ep_map.ep_num = nranks;
    team_params.ep_map.array.map = rank_mapping.data();
    team_params.ep_map.array.elem_size = sizeof(rank_mapping[0]);
    team_params.ep_range = UCC_COLLECTIVE_EP_RANGE_CONTIG;
    team_params.ep = rank;
    team_params.id = 0;
    UCC_CHECK(ucc_team_create_post(&ctx, 1, &team_params, &team));
    while (ucc_team_create_test(team) == UCC_INPROGRESS);
}

int create_ucc_comm(UCCComm &comm, int tid, int num_threads, const ucc_lib_h lib)
{
  int mpi_rank, mpi_size;
  MPI_Comm mpi_comm = MPI_COMM_WORLD;
  MPI_Comm_rank(mpi_comm, &mpi_rank);
  MPI_Comm_size(mpi_comm, &mpi_size);

  // creating communicator handle with MPI communicator
  int size = mpi_size * num_threads;
  int rank = num_threads * mpi_rank + tid;

  comm.mpi_comm_size = mpi_size;
  comm.mpi_rank = mpi_rank;
  comm.nb_threads = num_threads;
  comm.tid = tid;
  comm.global_rank = rank;
  comm.comm = MPI_COMM_WORLD;
  comm.global_comm_size = size;

  ucc_context_h ctx;
  ucc_team_h team;
  create_ucc_cxt(comm, lib, ctx);
  std::vector<int> top_rank_mapping;
  top_rank_mapping.resize(size);
  std::iota(top_rank_mapping.begin(), top_rank_mapping.end(), 0);
  create_ucc_team(rank, size, top_rank_mapping, ctx, team);

  comm.ctx = ctx;
  comm.team = team;
  return 0;
}

int destroy_ucc_comm(UCCComm &comm)
{
  while (ucc_team_destroy(comm.team) == UCC_INPROGRESS) {};
  ucc_context_destroy(comm.ctx);
  return 0;
}