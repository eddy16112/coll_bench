#include "util/util.h"
#include <ucc/api/ucc.h>
#include <stdio.h>
#include <assert.h>
#include <vector>
#include <iostream>

//#define _USE_UCC_EP_MAP

#define STR(x) #x
#define UCC_CHECK(_call)                                            \
  if (UCC_OK != (_call)) {                                          \
    std::cerr << "*** UCC TEST FAIL: " << STR(_call) << "\n";       \
    assert(0);                                                      \
  }

struct UCCComm {
  MPI_Comm comm;
  int mpi_comm_size;
  int nb_threads;
  int mpi_rank;
  int tid;
  int rank;
  int comm_size;
  int starting_tag = 0;
  ucc_context_h ctx;
  ucc_team_h team;
#ifdef _USE_UCC_EP_MAP
  std::vector<int> top_rank_mapping;
#endif
};

void create_ucc_lib(ucc_lib_h &lib);

void create_ucc_cxt(UCCComm &comm, ucc_lib_h lib, ucc_context_h &ctx);

void create_ucc_team(UCCComm &comm, std::vector<int> &rank_mapping, ucc_context_h ctx, ucc_team_h &team);

int create_ucc_comm(UCCComm &comm, int tid, int num_threads, const ucc_lib_h lib);

int destroy_ucc_comm(UCCComm &comm);

void ucc_allreduce(void *sendbuf, ucc_memory_type_t send_memtype, void *recvbuf, ucc_memory_type_t recv_memtype, size_t count, ucc_datatype_t type, ucc_reduction_op_t op, UCCComm &comm);

void ucc_barrier(UCCComm &comm);

void ucc_reduce(void *sendbuf, ucc_memory_type_t send_memtype, void *recvbuf, ucc_memory_type_t recv_memtype, size_t count, ucc_datatype_t type, ucc_reduction_op_t op, int root, UCCComm &comm);

ucc_datatype_t get_ucc_datatype(enum DataType datatype);

ucc_reduction_op_t get_ucc_op(enum Redop op);

size_t get_ucc_datatype_size(enum DataType datatype);

ucc_memory_type_t get_ucc_memtype(enum MemType memtype);