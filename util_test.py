import ray
from alpa.collective.collective_group.base_collective_group import BaseGroup, Rendezvous
from alpa import init
from alpa.device_mesh import (
    create_and_record_cross_mesh_collective_communicators, get_global_cluster)
from alpa.pipeline_parallel.stage_construction import get_sliced_virtual_submeshes

init('ray')

def _generate_nccl_uid():
    group_uid = [1,2,3,4,5,6]
    store_name = 's_name'
    print("store_name: ", store_name)
    # Avoid a potential circular dependency in ray/actor.py
    from alpa.collective.util import NCCLUniqueIDStore  # pylint: disable=import-outside-toplevel
    store = NCCLUniqueIDStore.options(
        name=store_name).remote(store_name)
    ray.get([store.set_id.remote(group_uid)])
    return group_uid


virtual_mesh = get_global_cluster().get_virtual_physical_mesh(host_ids=[0,1], num_devices_per_host=1)
# host_ids=[0], num_devices_per_host=2
# submesh_shapes = [(1, 1),(1, 1), (1, 1), (1, 1)] * 1
submesh_shapes = [(1, 1)] * 2
# submesh_shapes = [(1, 1),(1, 1)] * 1
sliced_virtual_meshes = get_sliced_virtual_submeshes(
    virtual_mesh, submesh_shapes)
virtual_mesh.get_physical_mesh_group(sliced_virtual_meshes)
mesh_group = virtual_mesh.launched_physical_mesh_group
meshes = mesh_group.meshes

nccl_uid=None
if rank == 0:
    nccl_uid = _generate_nccl_uid()
else:
    if nccl_uid is None:
        rendezvous = Rendezvous()
        rendezvous.meet(timeout_s=3000)
        nccl_uid = rendezvous.get_nccl_id()