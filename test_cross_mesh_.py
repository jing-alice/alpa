import os

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
w_size = comm.Get_size()

from jax._src.lib import xla_extension as xe
from jax.lib.xla_bridge import get_backend
print(rank, w_size, xe.nccl_get_version())

d= os.environ
d2={}
for i in d:
    if 'NCCL' in i:
        d2[i] = d[i]
print(f"{rank=},  {d2}")

if rank == 0:
    nccl_uid = xe.nccl_get_unique_id()
else:
    nccl_uid = None

nccl_uid = comm.bcast(nccl_uid)

print(type(nccl_uid),type(nccl_uid[0]),len(nccl_uid))
print(w_size, [rank], [0])
# xla_comm_group = xe.CommGroup(get_backend())
aa = xe.nccl_create_com(w_size,[rank], [0], nccl_uid)
print('finished',aa.ok())
