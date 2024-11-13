#include "util/util.h"
#include <ucc/api/ucc.h>
#include <stdio.h>
#include <assert.h>
#include <vector>
#include <iostream>

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
  int global_rank;
  int global_comm_size;
  int starting_tag = 0;
  ucc_context_h ctx;
  ucc_team_h team;
};

void create_ucc_lib(ucc_lib_h &lib);

void create_ucc_cxt(UCCComm &comm, ucc_lib_h lib, ucc_context_h &ctx);

void create_ucc_team(int rank, int nranks, std::vector<int> &rank_mapping, ucc_context_h ctx, ucc_team_h &team);

int create_ucc_comm(UCCComm &comm, int tid, int num_threads, const ucc_lib_h lib);

int destroy_ucc_comm(UCCComm &comm);