#include <stdio.h>
#include "cuda_runtime.h"
#include "nccl.h"
#include "mpi.h"
#include <unistd.h>
#include <stdint.h>
#include <stdlib.h>
#include <iostream>
#include <vector>

#define MPICHECK(cmd) do {                          \
  int e = cmd;                                      \
  if( e != MPI_SUCCESS ) {                          \
    printf("Failed: MPI error %s:%d '%d'\n",        \
        __FILE__,__LINE__, e);   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",             \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",             \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


using AlpaNcclUid = std::vector<int8_t>;
using AlpaUuids = std::vector<int>;

ncclUniqueId NcclUidDeserialize(const AlpaNcclUid &nccl_uid_vec) {
  ncclUniqueId nccl_uid;
  // CHECK_EQ(sizeof(nccl_uid.internal), nccl_uid_vec.size());
  memcpy(&nccl_uid.internal, nccl_uid_vec.data(), sizeof(nccl_uid.internal));
  return nccl_uid;
}
AlpaNcclUid NcclUidSerialize(ncclUniqueId nccl_uid) {
  AlpaNcclUid nccl_uid_vec(sizeof(nccl_uid.internal), 0);
  memcpy(nccl_uid_vec.data(), &nccl_uid.internal, sizeof(nccl_uid.internal));
  return nccl_uid_vec;
}
// Communicator related functions:
void NcclCreateCommunicators(
    int world_size, const std::vector<int> &device_global_ranks,
    const std::vector<int> &device_ids, const AlpaNcclUid &nccl_uid_vec) {

  int n_devices = device_global_ranks.size();
  std::cout << "XLA_devices: " << n_devices << std::endl;
  std::cout <<"world_size: "<<world_size<< "  n_devices: " << n_devices<<"  device_global_ranks[0]: "<< device_global_ranks[0]<< "  device_ids[0]: "<< device_ids[0] << std::endl;
  // CHECK_EQ(n_devices, device_ids.size());

  ncclUniqueId nccl_uid = NcclUidDeserialize(nccl_uid_vec);
  std::cout << "XLA_ncclUID " << std:: endl;

  // Create Communicators
  // XLA_CUDA_RETURN_IF_ERROR(ncclGroupStart());
  std::cout << "XLA_start " << std:: endl;

  for (int i = 0; i < n_devices; i++) {
    cudaSetDevice(device_ids[i]);
    int rank = device_global_ranks[i];
    // auto comm_key = std::make_pair(nccl_uid_vec, device_ids[i]);
    // NcclComm::Lock comm = comm_map[comm_key].Acquire();
    ncclComm_t comm;
    NCCLCHECK(
        ncclCommInitRank(&comm, world_size, nccl_uid, rank));
  }
  std::cout << "XLA_fnish_Init " << std::endl;

  // local_ids[nccl_uid_vec] = device_ids;
  // XLA_CUDA_RETURN_IF_ERROR(ncclGroupEnd());
  std::cout << "XLA_NCCL_end " << std::endl;
}

void NcclCreateCommunicators2(
    int world_size, const std::vector<int> &device_global_ranks,
    const std::vector<int> &device_ids, const AlpaNcclUid &nccl_uid_vec) {

  int n_devices = device_global_ranks.size();
  std::cout <<"world_size: "<<world_size<< "  n_devices: " << n_devices<<"  device_global_ranks[0]: "<< device_global_ranks[0]<< "  device_ids[0]: "<< device_ids[0] << std::endl;
  // CHECK_EQ(n_devices, device_ids.size());

  ncclUniqueId nccl_uid = NcclUidDeserialize(nccl_uid_vec);
  std::cout << "XLA_ncclUID " << std:: endl;
  std::cout<<"\nnccl_uid.internal:  ";
  int total = 0;
  for(int i=0; i<128;i++){
    std::cout<<(int)nccl_uid.internal[i]<<',';
    total+=nccl_uid.internal[i];
  }
  std::cout<<"  toatl: "<<total<<'\n';

  // Create Communicators
  // XLA_CUDA_RETURN_IF_ERROR(ncclGroupStart());
  std::cout << "XLA_start " << std:: endl;

  for (int i = 0; i < n_devices; i++) {
    cudaSetDevice(device_ids[i]);
    int rank = device_global_ranks[i];
    // auto comm_key = std::make_pair(nccl_uid_vec, device_ids[i]);
    // NcclComm::Lock comm = comm_map[comm_key].Acquire();
    ncclComm_t comm;
    NCCLCHECK(
        ncclCommInitRank(&comm, world_size, nccl_uid, rank));
  }
  std::cout << "XLA_fnish_Init " << std::endl;

  // local_ids[nccl_uid_vec] = device_ids;
  // XLA_CUDA_RETURN_IF_ERROR(ncclGroupEnd());
  std::cout << "XLA_NCCL_end " << std::endl;
  // return OkStatus();
}

union cv{
  int a;
  char b[4];
};

int main(int argc, char* argv[])
{
  int myRank, nRanks, localRank = 0;

  //initializing MPI
  MPICHECK(MPI_Init(&argc, &argv));
  MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &myRank));
  MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &nRanks));
  int tmp = myRank%2;
  
  std::cout<<"nRanks: "<<nRanks<< ",  myRank: "<< myRank << ", tmp: "<<tmp <<"\n";

  ncclUniqueId id;
  ncclComm_t comm;


  //get NCCL unique ID at rank 0 and broadcast it to all others
  if (myRank == 0) ncclGetUniqueId(&id);
  MPICHECK(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));
  cv ccv;
  ccv.a=0;
  for(int i=0; i<128;i++){
    ccv.b[0] = id.internal[i];
    std::cout<< ccv.a <<',';
  }
  std::cout<<'\n';



  int world_size = nRanks;
  std::vector<int> device_global_ranks{myRank};
  std::vector<int> device_ids{0};
  AlpaNcclUid nccl_uid_vec = NcclUidSerialize(id);
  // AlpaNcclUid nccl_uid_vec{-54,126,6,105,-40,82,-12,-13,2,0,-84,-125,10,-23,111,-56,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,114,24,2,82,-60,44,-121,1,0,0,0,0,0,0,0,32,114,12,70,62,86,0,0,0,20,-67,68,62,86,0,0,64,-40,24,-96,-2,127,0,0,-40,-39,24,-96,-2,127,0,0,32,-60,-76,68,62,86,0,0,-24,-78,0,-54,117,127,0,0,39,14,-26,-55,117,127,0,0,0,0,0,0,0,0,0,0,8,0,-67,68,62,86,0,0};
  std::cout<<nccl_uid_vec.size()<<std::endl;
  NcclCreateCommunicators2(world_size, device_global_ranks, device_ids, nccl_uid_vec);

  //initializing NCCL
  // ncclGroupStart();
  // for(int i=0;i<1;i++){
  //   cudaSetDevice(i);
  //   NCCLCHECK(ncclCommInitRank(&comm, nRanks, id, myRank));
  // }
  // ncclGroupEnd();

  MPICHECK(MPI_Finalize());

  printf("[MPI Rank %d] Success \n", myRank);
  return 0;
}